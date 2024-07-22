#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
# 导入Sail模块和其他必要的库
#写中文注释
import sophon.sail as sail
from transformers import AutoTokenizer
import numpy as np
import time
import argparse
import pdb

# 根据Sail数据类型转换为NumPy数据类型
#convert sail_dtype to numpy dtype
def type_convert(sail_dtype):
    """
    根据Sail数据类型转换为NumPy数据类型。
    
    参数:
    sail_dtype -- Sail数据类型。
    
    返回:
    NumPy对应的data type。
    """
    if sail_dtype == sail.Dtype.BM_FLOAT32:
        return np.float32
    if sail_dtype == sail.Dtype.BM_FLOAT16:
        return np.float16
    if sail_dtype == sail.Dtype.BM_INT32:
        return np.int32
    if sail_dtype == sail.Dtype.BM_BFLOAT16: # 后续需要修改bf16的接口,现在先用fp16的代替
        return np.float16
    
    raise TypeError("only support float32 and int32 right now")

# 将float16数组重新解释为uint16数组，因为pybind11不支持float16
def fp16_cast(arr:np.ndarray): #这个接口的作用在于把np.float16假冒成np.uint16传进Tensor，sail update_data如果能接收传输二进制，那就不需要这个了。(后续需要改成bf16的接口)
    """
    将float16数组重新解释为uint16数组，因为pybind11不支持float16。

    参数:
    arr -- 输入的NumPy数组。

    返回:
    重新解释为uint16的数组。
    """
    if arr.dtype == np.float16:
        return arr.view(np.uint16)
    else:
        return arr

# Qwen1_5类用于封装Sail引擎的推理过程
class Qwen1_5:
    """
    Qwen1_5类用于封装Sail引擎的推理过程，包括初始化和前向传播。
    
    参数:
    handle -- Sail的句柄。
    engine -- Sail的引擎。
    tokenizer -- 用于 token 化的工具。
    """
    def __init__(self, handle, engine, tokenizer):
        self.net = engine
        self.tokenizer = tokenizer
        self.handle = handle
        # 获取网络的图名和设备ID
        self.graph_names = self.net.get_graph_names()
        self.dev_id = self.net.get_device_ids()[0]

        # 从tokenizer获取结束符ID和网络的层数、序列长度和隐藏层大小
        self.EOS = self.tokenizer.eos_token_id
        self.NUM_LAYERS = (len(self.graph_names) - 2) // 2
        _, self.SEQLEN, self.HIDDEN_SIZE = self.first_hidden_input_shape = self.net.get_input_shape("block_0", 0)

        # 初始化推理过程中的变量和参数
        self.is_greedy_sample = False
        self.name_embed = "embedding"
        self.name_embed_cache = "embedding_cache"
        self.name_lm = "lm_head"
        self.name_blocks = ["block_"+str(i) for i in range(self.NUM_LAYERS)]
        self.name_blocks_cache = ["block_cache_"+str(i) for i in range(self.NUM_LAYERS)]
        self.name_sample = "greedy_head" if self.is_greedy_sample else "penalty_sample_head"

        # 初始化第一步和下一步的嵌入输入和输出
        self.first_embed_input = self.init_sail_tensor(self.name_embed, 0, [1, self.SEQLEN])
        self.first_embed_output = self.init_sail_tensor(self.name_embed, 0, [1, self.SEQLEN, self.HIDDEN_SIZE], False)
        
        self.next_embed_input = self.init_sail_tensor(self.name_embed_cache, 0, [1, 1])
        self.next_embed_output = self.init_sail_tensor(self.name_embed_cache, 0, [1,  self.HIDDEN_SIZE], False)

        # 初始化第一步和下一步的隐藏状态输入和输出
        self.first_hidden_input = self.init_sail_tensor(self.name_blocks[0], 0)
        self.first_hidden_output = self.init_sail_tensor(self.name_blocks[0], 0, None, False)

        self.next_hidden_input = self.init_sail_tensor(self.name_blocks_cache[0], 0)
        self.next_hidden_output = self.init_sail_tensor(self.name_blocks_cache[0], 0, None, False)

        # 初始化第一步和下一步的位置ID和注意力掩码
        self.first_pid = self.init_sail_tensor(self.name_blocks[0], 1)
        self.first_attention = self.init_sail_tensor(self.name_blocks[0], 2)
       
        self.next_pid = self.init_sail_tensor(self.name_blocks_cache[0], 1)
        self.next_attention = self.init_sail_tensor(self.name_blocks_cache[0], 2)

        # 初始化KV缓存
        self.present_key = self.init_sail_tensor(self.name_blocks_cache[0], 1, None, False)
        self.present_value = self.init_sail_tensor(self.name_blocks_cache[0], 2, None, False)

        # 初始化past key和value的输出
        self.past_key_output = []
        self.past_value_output = []

        for _ in range(self.NUM_LAYERS):
            
            self.past_key_output.append(self.init_sail_tensor(self.name_blocks[0], 1, None, False))
            self.past_value_output.append(self.init_sail_tensor(self.name_blocks[0], 2, None, False))

        # 初始化lm_head的输入和输出
        self.lm_input = self.init_sail_tensor(self.name_lm, 0)
        self.lm_output = self.init_sail_tensor(self.name_lm, 0, None, False)

        # 初始化sample的输入和输出（目前未使用）
        # self.sample_input = self.init_sail_tensor(self.name_sample, 0)
        # self.sample_output = self.init_sail_tensor(self.name_sample, 0, None, False)


    def init_sail_tensor(self, name, tensor_idx, shape=None, is_input=True):
        """
        初始化Sail张量。
        
        参数:
        name -- 图或网络的名称。
        tensor_idx -- 输入或输出张量的索引。
        shape -- 张量的形状。
        is_input -- 张量是否为输入张量。
        
        返回:
        """
        """
        init a sail tensor of sail.engine.
        parameters:
        input:
            name: str, graph_name/net_name
            tensor_idx: int, input/output tensor id
            shape: list[int], shape of tensor
            is_input: bool, is input tensor or not
        return:
            dict
        """
        tensor = {}
        
        if is_input:
            print('init input name is :', name)
            tensor["name"] = self.net.get_input_names(name)[tensor_idx]
            tensor["shape"] = self.net.get_input_shape(name, tensor_idx) if shape is None else shape
            tensor["dtype"] = self.net.get_input_dtype(name, tensor_idx)
            tensor["data"] = sail.Tensor(self.handle, tensor["shape"], tensor["dtype"], False, True)
            print('---------is input tensor data-----------')
        else:
            print('init output name is :', name)
            tensor["name"] = self.net.get_output_names(name)[tensor_idx]
            tensor["shape"] = self.net.get_output_shape(name, tensor_idx) if shape is None else shape
            tensor["dtype"] = self.net.get_output_dtype(name, tensor_idx)
            tensor["data"] = sail.Tensor(self.handle, tensor["shape"], tensor["dtype"], False, True)
            print('************output tensor data*********')
        return tensor


    def forward_first(self, token):
        """
        初始化模型的前向传播过程，处理第一个token。

        参数:
        - token: 输入的token序列。

        返回:
        - lm_output的整数numpy数组。
        """
        # 记录开始时间
        # Keep
        test_time1 = time.time()
        
        # 初始化输入id矩阵，并填充token
        input_ids = np.zeros(self.SEQLEN, type_convert(self.first_embed_input["dtype"]))
        input_ids[:min(self.SEQLEN, len(token))] = token
        input_ids = input_ids.reshape(1, -1)
        
        # 记录token长度
        self.token_length = len(token)
        
        # 初始化位置id矩阵
        position_id = np.zeros(self.SEQLEN, type_convert(self.first_pid["dtype"])) 
        for i in range(self.token_length):
            position_id[i] = i
        
        # 记录中间时间点
        test_time2 = time.time()
        # 输出第一阶段耗时
        print('first stage',test_time2-test_time1)
        
        # 初始化注意力掩码
        attention_mask = np.ones(self.SEQLEN*self.SEQLEN, type_convert(self.first_attention["dtype"])) * (-10000.0)
        for i in range(self.token_length):
            for j in range(self.SEQLEN):
                if (j <= i):
                    attention_mask[i*self.SEQLEN + j] = 0
        
        # 记录中间时间点
        test_time3 = time.time()
        # 输出第二阶段耗时
        print('second stage',test_time3-test_time2)
        
        # 更新嵌入输入数据
        self.first_embed_input["data"].update_data(input_ids)
        input_embed_tensors = {0: self.first_embed_input["data"]}
        output_embed_tensors = {0: self.first_embed_output["data"]}
        self.net.process(self.name_embed, input_embed_tensors, output_embed_tensors)
        
        # 记录中间时间点
        test_time4 = time.time()
        # 输出第四阶段耗时
        print('4-3',test_time4-test_time3)
        
        # 处理隐藏层状态和位置id
        self.first_hidden_tensor = self.first_embed_output["data"]
        self.first_hidden_tensor.reshape(self.first_hidden_input["shape"])
        self.first_pid["data"].update_data(position_id.reshape(self.first_pid["shape"]))
        self.first_attention["data"].update_data(fp16_cast(attention_mask.reshape(self.first_attention["shape"])))
        
        # 准备块（block）的输入数据
        input_blocks_tensors = {0: self.first_hidden_tensor, 
                                1: self.first_pid["data"], 
                                2: self.first_attention["data"]}
        test_time7 = time.time()
        for i in range(self.NUM_LAYERS):
            # 处理每个层的块
            output_blocks_tensors = {0: self.first_hidden_tensor,
                                    1: self.past_key_output[i]["data"],
                                    2: self.past_value_output[i]["data"],}
            self.net.process(self.name_blocks[i], input_blocks_tensors, output_blocks_tensors)
        
        # 记录中间时间点
        test_time5 = time.time()
        # 输出第五阶段耗时
        print('5-4',test_time5-test_time4)
        # 输出整个第七阶段耗时
        print('7-4',test_time7-test_time4)
        
        # 准备语言模型（lm）的输入数据
        # lm_head
        # hidden_states 的最后一个位置的元素取出来作为 lm_head的输入
        copy_len = self.first_hidden_tensor.shape()[-1]
        self.lm_input["data"].sync_d2d(self.first_hidden_tensor,
                                      (self.token_length-1)* copy_len,  
                                      0, 
                                      copy_len)
        
        input_lm_tensors = {0: self.lm_input["data"]}
        output_lm_tensors = {0: self.lm_output["data"]}

        # 处理语言模型层
        self.net.process(self.name_lm, input_lm_tensors, output_lm_tensors)
        
        # 记录中间时间点
        test_time6 = time.time()
        # 输出第六阶段耗时
        print('6-5',test_time6-test_time5)
        
        # 返回语言模型的输出
        return int(self.lm_output["data"].asnumpy())

    def forward_next(self, ):
        """
        继续模型的前向传播过程，处理下一个token。

        返回:
        - lm_output的整数numpy数组。
        """
        # 初始化注意力掩码
        attention_mask = np.zeros(self.SEQLEN+1, type_convert(self.next_attention["dtype"]))
        for i in range(self.token_length-1, self.SEQLEN):
            attention_mask[i] = -10000.0
        
        # 初始化位置id
        position_id = np.array(self.token_length - 1, type_convert(self.next_pid["dtype"]))

        # 更新嵌入输入数据
        self.next_embed_input["data"] = self.lm_output["data"]
        self.next_embed_input["data"].reshape(self.next_embed_input["shape"])

        input_embed_tensors = {0: self.next_embed_input["data"]}
        output_embed_tensors = {0: self.next_embed_output["data"]}
        # Embedding Layer Inference
        self.net.process(self.name_embed_cache, input_embed_tensors, output_embed_tensors)

        # 更新位置id和注意力掩码数据
        self.next_pid["data"].update_data(position_id.reshape(self.next_pid["shape"]))
        self.next_attention["data"].update_data(fp16_cast(attention_mask.reshape(self.next_attention["shape"])))

        self.next_hidden_tensor = self.next_embed_output["data"]
        self.next_hidden_tensor.reshape(self.next_hidden_input["shape"])

        # 处理每个层的块
        for i in range(self.NUM_LAYERS):
            inputs_block_cache_tensors = {0: self.next_hidden_tensor, 
                                        1: self.next_pid["data"], 
                                        2: self.next_attention["data"], 
                                        3: self.past_key_output[i]["data"], 
                                        4: self.past_value_output[i]["data"]}
            outputs_block_cache_tensors = {0: self.next_hidden_tensor,
                                        1: self.present_key["data"],
                                        2: self.present_value["data"]}
            self.net.process(self.name_blocks_cache[i], inputs_block_cache_tensors, outputs_block_cache_tensors)

            # 更新KV缓存
            unit_size = self.present_key["shape"][-1]*self.present_key["shape"][-2]
            self.past_key_output[i]["data"].sync_d2d(self.present_key["data"], 0, (self.token_length-1)*unit_size, unit_size)
            self.past_value_output[i]["data"].sync_d2d(self.present_value["data"], 0, (self.token_length-1)*unit_size, unit_size)

        self.lm_input_tensor = self.next_hidden_tensor
        self.lm_input_tensor.reshape(self.lm_input["shape"])
        
        input_lm_tensors = {0: self.lm_input_tensor}
        output_lm_tensors = {0: self.lm_output["data"]}
        
        # 处理语言模型层
        # Lm_head Inference
        self.net.process(self.name_lm, input_lm_tensors, output_lm_tensors)

        # 返回语言模型的输出
        return int(self.lm_output["data"].asnumpy())

    def chat_stream(self, input, history):
        """
        生成对话流。
        
        通过不断调整输入历史记录的长度，确保模型输入不超过预设的最大长度限制。
        一旦生成的回应达到最大长度或遇到结束标记，对话流生成结束。
        
        参数:
        - input: 用户的输入字符串。
        - history: 之前的对话历史列表。
        
        生成值:
        - 每次模型生成的单个令牌（token），直到达到最大长度或生成结束标记。
        """
        # 初始化输入历史记录
        input_history = [{"role": "user", "content": input}]
        # 根据输入生成模板化的文本
        input_text = self.tokenizer.apply_chat_template(input_history, tokenize=False, add_generation_prompt=True)
        # 将文本转换为令牌（token）序列
        input_tokens = self.tokenizer(input_text).input_ids
        # 如果输入令牌序列过长，则直接返回提示信息
        if (len(input_tokens) > self.SEQLEN / 3):
            yield '##INPUT_TOO_LONG'
            return

        # 更新对话历史
        history.append({"role": "user", "content": input})
        # 生成包含整个对话历史的模板化文本
        text = self.tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        # 将文本转换为令牌序列
        tokens = self.tokenizer(text).input_ids
        # 如果令牌序列超过一半的最长长度，则通过移除早期历史来缩短序列
        while (len(tokens) > self.SEQLEN / 2):
            history = history[1:]
            text = self.tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
            tokens = self.tokenizer(text).input_ids
        # 初始化令牌生成计数器和时间记录
        tok_num = 0
        first_start = time.time()
        # 开始首次令牌生成
        token = self.forward_first(tokens)
        first_end = time.time()
        # 生成令牌直到达到最大长度或遇到结束标记
        while token != self.EOS and self.token_length < self.SEQLEN:
            # 将生成的令牌转换为字符串并yield出去
            diff = self.tokenizer.decode([token])
            yield diff
            # 更新已生成令牌长度
            if self.token_length < self.SEQLEN:
                self.token_length += 1
            tok_num += 1
            # 继续生成下一个令牌
            token = self.forward_next()
        
        # 如果生成的令牌长度达到最大值，则提示并返回
        if self.token_length >= self.SEQLEN:
            yield '##TOKEN_LENGTH_MAX'
            return
        
        # 记录最后一次生成的时间和生成速率
        next_end = time.time()
        first_duration = first_end-first_start
        next_duration = next_end-first_end
        tps = tok_num / next_duration
        # 打印首次生成和后续生成的时间及生成速率信息
        print('\n\n')
        print(f"FTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")
   
# 定义一个应用程序接口，用于与用户进行交互
def app(client):
    # 初始化会话历史记录
    history = []
    # 进入无限循环，直到用户选择退出
    while True:
        # 获取用户输入的问题
        input_str = input("\nQuestion: ")
        # 检查是否要退出应用程序
        if input_str == "exit":
            break
        # 输出回答前的提示
        print("\nAnswer: ")
        # 初始化助手的回答消息
        assistant_msg = ''
        # 通过客户端的聊天流接口处理用户的输入，并实时输出部分回答
        for response in client.chat_stream(input_str, history):
            assistant_msg = response
            print(response, flush=True, end='')
        # 将用户的提问和助手的回答添加到会话历史中
        history.append({"role": "user", "content": input_str})
        history.append({"role": "assistant", "content": assistant_msg})

# 主函数，用于程序的初始化和启动交互式会话
def main(args):
    # 初始化sail处理句柄
    handle = sail.Handle(args.dev_id)
    # 初始化自动分词器
    tokenizer = AutoTokenizer.from_pretrained(args.token, trust_remote_code=True)
    # 初始化语言模型引擎
    engine = sail.EngineLLM(args.bmodel, [args.dev_id])
    # 初始化客户端，用于与用户交互
    # breakpoint()
    client = Qwen1_5(handle, engine, tokenizer)
    # 启动应用程序
    app(client)

# 定义命令行参数解析函数，用于获取运行时参数
def argsparser():
    # 初始化参数解析器
    parser = argparse.ArgumentParser(prog=__file__)
    # 添加模型路径参数
    parser.add_argument('--bmodel', type=str, default='./qwen1.5-7b_int4_1dev.bmodel', help='path of bmodel')
    # 添加分词器路径参数
    parser.add_argument('--token', type=str, default='./python/token_config/', help='path of tokenizer')
    # 添加设备ID参数
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    # 解析命令行参数并返回
    args = parser.parse_args()
    return args

# 程序入口
if __name__ == "__main__":
    # 解析命令行参数
    args = argsparser()
    # 初始化并启动应用程序
    main(args)
    # 程序运行结束提示
    print('all done')