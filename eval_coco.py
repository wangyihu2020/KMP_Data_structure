#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import argparse
import json
import logging
logging.basicConfig(level=logging.DEBUG)

# 引入COCO数据集处理工具
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 定义命令行参数解析函数
# 中文注释
def argsparser():
    """
    解析命令行参数。
    
    返回:
    args: 解析后的命令行参数。
    """
    parser = argparse.ArgumentParser(prog=__file__)
    # 指定标注文件路径的参数
    parser.add_argument('--gt_path', type=str, default='../datasets/coco/instances_val2017.json', help='path of label json')
    # 指定结果JSON文件路径的参数
    parser.add_argument('--result_json', type=str, default='../python/results/yolov8s_fp32_1b.bmodel_val2017_bmcv_python_result.json', help='path of result json')
    # 指定评估类型的参数
    parser.add_argument('--ann_type', type=str, default='bbox', help='type of evaluation')
    args = parser.parse_args()
    return args

# 定义将COCO80类映射到COCO91类的函数
def coco80_to_coco91_class():
    """
    将COCO 2014年的80个类别ID映射到COCO 2017年的91个类别ID。
    
    返回:
    类别ID的列表，按照COCO 2014年的顺序。
    """
    # 这里返回的是一个列表，列表中的每个元素是COCO 2014年类别ID在COCO 2017年中的新ID
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
         21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
         41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
         59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
         80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x
    
# 定义转换关键点格式的函数
def convert_to_coco_keypoints(json_file, cocoGt):
    """
    将OpenPose格式的关键点数据转换为COCO格式。
    
    参数:
    json_file: 包含OpenPose格式关键点数据的JSON文件路径。
    cocoGt: COCOGroundtruth对象，用于获取图像和类别信息。
    """
    # OpenPose关键点的名称，与COCO关键点名称对应
    keypoints_openpose_map = ["nose", "Neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow", "left_wrist",  \
                               "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye", "left_eye", "right_ear", "left_ear"]
    temp_json = []
    images_list = cocoGt.dataset["images"]
    with open(json_file, 'r') as f:
        res_json = json.load(f)
    for res in res_json:
        image_name = res["image_name"]
        keypoints = res["keypoints"]
        if len(keypoints) == 0:
            continue
        for image in images_list:
            if image_name == image["file_name"]:
                image_id = image["id"]
                break
        person_num = int(len(keypoints) / (len(keypoints_openpose_map) * 3))
        for i in range(person_num):
            data = dict()
            data['image_id'] = int(image_id)
            data['category_id'] = 1
            data['keypoints'] = []
            score_list = []
            for point_name in cocoGt.dataset["categories"][0]["keypoints"]:
                point_id = keypoints_openpose_map.index(point_name)
                x = keypoints[(i * len(keypoints_openpose_map) + point_id) * 3]
                y = keypoints[(i * len(keypoints_openpose_map) + point_id) * 3 + 1]
                score = keypoints[(i * len(keypoints_openpose_map) + point_id) * 3 + 2]
                data['keypoints'].append(x)
                data['keypoints'].append(y)
                data['keypoints'].append(score)
                score_list.append(score)
            data['score'] = float(sum(score_list)/len(score_list) + 1.25 * max(score_list))
            temp_json.append(data)
    with open('temp.json', 'w') as fid:
        json.dump(temp_json, fid)

# 定义转换边界框格式的函数
def convert_to_coco_bbox(json_file, cocoGt):
    """
    将预测的边界框数据转换为COCO格式。
    
    参数:
    json_file: 包含预测边界框数据的JSON文件路径。
    cocoGt: COCOGroundtruth对象，用于获取图像和类别信息。
    """
    temp_json = []
    coco91class = coco80_to_coco91_class()
    images_list = cocoGt.dataset["images"]
    with open(json_file, 'r') as f:
        res_json = json.load(f)
    for res in res_json:
        image_name = res["image_name"]
        bboxes = res["bboxes"]
        if len(bboxes) == 0:
            continue
        for image in images_list:
            if image_name == image["file_name"]:
                image_id = image["id"]
                break
            
        for i in range(len(bboxes)):
            data = dict()
            data['image_id'] = int(image_id)
            data['category_id'] = coco91class[bboxes[i]['category_id']]
            data['bbox'] = bboxes[i]['bbox']
            data['score'] = bboxes[i]['score']
            temp_json.append(data)
            
    with open('temp.json', 'w') as fid:
        json.dump(temp_json, fid)

# 主函数
def main(args):
    """
    主函数，执行COCO格式数据的转换和评估。
    
    参数:
    args: 命令行参数对象，包含输入输出文件路径和评估类型。
    """
    cocoGt = COCO(args.gt_path)
    if args.ann_type == 'keypoints':
        convert_to_coco_keypoints(args.result_json, cocoGt)
    if args.ann_type == 'bbox':
        convert_to_coco_bbox(args.result_json, cocoGt)
    
    cocoDt = cocoGt.loadRes('temp.json')
    cocoEval = COCOeval(cocoGt, cocoDt, args.ann_type)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    logging.info("mAP = {}".format(cocoEval.stats[0]))

if __name__ == '__main__':
    args = argsparser()
    main(args)