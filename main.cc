//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-STREAM is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <functional>

#include "draw_funcs.h"
#include <termios.h>

using namespace std;

typedef struct demo_config_ {
  int num_graphs;
  int num_channels_per_graph;
  std::vector<nlohmann::json> channel_configs;
  nlohmann::json report_config;
  nlohmann::json listen_config;
  bool download_image;
  std::string engine_config_file;
  std::string cameras_config_file;  // add by ccy
  std::vector<std::string> class_names;
  std::string draw_func_name;
  std::vector<std::string> car_attr;
  std::vector<std::string> person_attr;
  std::string heatmap_loss;
} demo_config;

constexpr const char* JSON_CONFIG_DOWNLOAD_IMAGE_FILED = "download_image";
constexpr const char* JSON_CONFIG_ENGINE_CONFIG_PATH_FILED =
    "engine_config_path";
// add by ccy
constexpr const char* JSON_CONFIG_CAMERAS_CONFIG_PATH_FILED =
    "cameras_config_path";
constexpr const char* JSON_CONFIG_CLASS_NAMES_FILED = "class_names";
constexpr const char* JSON_CONFIG_CHANNEL_CONFIG_FILED = "channels";
constexpr const char* JSON_CONFIG_CHANNEL_CONFIG_CHANNEL_ID_FILED =
    "channel_id";
constexpr const char* JSON_CONFIG_CHANNEL_CONFIG_URL_FILED = "url";
constexpr const char* JSON_CONFIG_CHANNEL_CONFIG_SOURCE_TYPE_FILED =
    "source_type";
constexpr const char* JSON_CONFIG_CHANNEL_CONFIG_LOOP_NUM_FILED = "loop_num";
constexpr const char* JSON_CONFIG_CHANNEL_CONFIG_FPS_FILED = "fps";
constexpr const char* JSON_CONFIG_CHANNEL_CONFIG_SAMPLE_INTERVAL_FILED =
    "sample_interval";
constexpr const char* JSON_CONFIG_CHANNEL_CONFIG_SKIP_ELEMENT_FILED =
    "skip_element";
constexpr const char* JSON_CONFIG_CHANNEL_CONFIG_SAMPLE_STRATEGY_FILED =
    "sample_strategy";
constexpr const char* JSON_CONFIG_CHANNEL_CONFIG_ROI_FILED = "roi";

constexpr const char* JSON_CONFIG_DRAW_FUNC_NAME_FILED = "draw_func_name";
constexpr const char* JSON_CONFIG_CAR_ATTRIBUTES_FILED = "car_attributes";
constexpr const char* JSON_CONFIG_PERSON_ATTRIBUTES_FILED = "person_attributes";
constexpr const char* JSON_CONFIG_CHANNEL_DECODE_IDX_FILED = "decode_id";
constexpr const char* JSON_CONFIG_HEATMAP_LOSS_CONFIG_FILED = "heatmap_loss";
constexpr const char* JSON_CONFIG_HTTP_REPORT_CONFIG_FILED = "http_report";
constexpr const char* JSON_CONFIG_HTTP_LISTEN_CONFIG_FILED = "http_listen";
constexpr const char* JSON_CONFIG_HTTP_CONFIG_IP_FILED = "ip";
constexpr const char* JSON_CONFIG_HTTP_CONFIG_PORT_FILED = "port";
constexpr const char* JSON_CONFIG_HTTP_CONFIG_PATH_FILED = "path";

// add by ccy
// parse cameralist
enum cameraStatus
{
  cameraStatus_unused,
  cameraStatus_used
};

typedef struct camera_config_{
  int channel_id;
  int camera_id;
  string url;
  string source_type;
  cameraStatus status;

} camera_config;

typedef struct camera_list_config_ {
  int num_cameras;
  std::vector<camera_config> camera_configs;

  void reset()
  {
    num_cameras = 0;
    camera_configs.clear();
  };
} camera_list_config;

struct camera_list_perChannel
{
  int channel_id;
  vector<string> urls;
  int current_url_index;

  void reset()
  {
    channel_id = 0;
    current_url_index = 0;
    urls.clear();
  }
};

typedef struct camera_sorted_info_{
  vector<camera_list_perChannel> channels;

  void reset()
  {
    channels.clear();
  }

}camera_sorted_info;

constexpr const char* JSON_CAMERAS_CONFIG_CHANNEL_CONFIG_FILED = "cameras";
constexpr const char* JSON_CAMERAS_CONFIG_CHANNEL_CONFIG_CHANNEL_ID_FILED =
    "channel_id";
constexpr const char* JSON_CAMERAS_CONFIG_CHANNEL_CONFIG_URL_FILED = "url";
constexpr const char* JSON_CAMERAS_CONFIG_CHANNEL_CONFIG_SOURCE_TYPE_FILED = 
    "source_type";
constexpr const char* JSON_CAMERAS_CONFIG_CHANNEL_CONFIG_CAMERA_ID_FILED =
    "camera_id";
constexpr const char* JSON_CAMERAS_CONFIG_CHANNEL_CONFIG_STATUS_FILED = "status";

void parse_camera_list_json(std::string& json_path, camera_list_config &config)
{
  std::ifstream istream;
  istream.open(json_path);
  STREAM_CHECK(istream.is_open(), "Please check camera config file ", json_path,
               " exists.");
  nlohmann::json cameras_json;
  istream >> cameras_json;
  istream.close();

  config.reset();

  auto camera_config_it = cameras_json.find(JSON_CAMERAS_CONFIG_CHANNEL_CONFIG_FILED);
  for (auto& channel_it : *camera_config_it) {
    camera_config cameraInfo;
    cameraInfo.channel_id =
        channel_it.find(JSON_CAMERAS_CONFIG_CHANNEL_CONFIG_CHANNEL_ID_FILED)
            ->get<int>();
    cameraInfo.url = channel_it.find(JSON_CAMERAS_CONFIG_CHANNEL_CONFIG_URL_FILED)
                              ->get<std::string>();
    cameraInfo.source_type =
        channel_it.find(JSON_CAMERAS_CONFIG_CHANNEL_CONFIG_SOURCE_TYPE_FILED)
            ->get<std::string>();
    cameraInfo.camera_id =
        channel_it.find(JSON_CAMERAS_CONFIG_CHANNEL_CONFIG_CAMERA_ID_FILED)
            ->get<int>();
    cameraInfo.status =
        channel_it.find(JSON_CAMERAS_CONFIG_CHANNEL_CONFIG_STATUS_FILED)
            ->get<std::string>() == "UNUSED"
            ? cameraStatus_unused
            : cameraStatus_used;

    config.camera_configs.push_back(cameraInfo);
    config.num_cameras ++;
  }

}

void sort_camera_list_info(camera_list_config config, camera_sorted_info_ &sorted_info)
{
  sorted_info.reset();

  for(int i = 0; i < config.num_cameras; i ++)
  {

    bool bFindChannel = false;
    for(int j = 0; j < sorted_info.channels.size(); j ++)
    {
      if(sorted_info.channels.at(j).channel_id == config.camera_configs.at(i).channel_id)
      {
        sorted_info.channels.at(j).urls.push_back(config.camera_configs.at(i).url);
        bFindChannel = true;
        break;
      }
    }
    
    if(!bFindChannel)
    {
      camera_list_perChannel perChannel;
      perChannel.reset();
      perChannel.channel_id = config.camera_configs.at(i).channel_id;
      perChannel.urls.push_back(config.camera_configs.at(i).url);

      sorted_info.channels.push_back(perChannel);
    }
  }
}
// add end

demo_config parse_demo_json(std::string& json_path) {
  std::ifstream istream;
  istream.open(json_path);
  STREAM_CHECK(istream.is_open(), "Please check config file ", json_path,
               " exists.");
  nlohmann::json demo_json;
  istream >> demo_json;
  istream.close();

  demo_config config;
  auto channel_config_it = demo_json.find(JSON_CONFIG_CHANNEL_CONFIG_FILED);

  config.download_image = false;
  if (demo_json.contains(JSON_CONFIG_DOWNLOAD_IMAGE_FILED))
    config.download_image =
        demo_json.find(JSON_CONFIG_DOWNLOAD_IMAGE_FILED)->get<bool>();
  config.engine_config_file =
      demo_json.find(JSON_CONFIG_ENGINE_CONFIG_PATH_FILED)->get<std::string>();
  // add by ccy
  if(demo_json.contains(JSON_CONFIG_CAMERAS_CONFIG_PATH_FILED))
  { 
    std::cout << "ccy find cameralist" << std::endl;
    config.cameras_config_file =
      demo_json.find(JSON_CONFIG_CAMERAS_CONFIG_PATH_FILED)->get<std::string>();
  }
  std::string class_names_file;
  if (demo_json.contains(JSON_CONFIG_CLASS_NAMES_FILED))
    class_names_file =
        demo_json.find(JSON_CONFIG_CLASS_NAMES_FILED)->get<std::string>();
  if (demo_json.contains(JSON_CONFIG_DRAW_FUNC_NAME_FILED))
    config.draw_func_name =
        demo_json.find(JSON_CONFIG_DRAW_FUNC_NAME_FILED)->get<std::string>();
  else
    config.draw_func_name = "default";
  std::string car_attr_file;
  if (demo_json.contains(JSON_CONFIG_CAR_ATTRIBUTES_FILED))
    car_attr_file =
        demo_json.find(JSON_CONFIG_CAR_ATTRIBUTES_FILED)->get<std::string>();
  std::string person_attr_file;
  if (demo_json.contains(JSON_CONFIG_PERSON_ATTRIBUTES_FILED))
    person_attr_file =
        demo_json.find(JSON_CONFIG_PERSON_ATTRIBUTES_FILED)->get<std::string>();
  if (demo_json.contains(JSON_CONFIG_HEATMAP_LOSS_CONFIG_FILED))
    config.heatmap_loss = demo_json.find(JSON_CONFIG_HEATMAP_LOSS_CONFIG_FILED)
                              ->get<std::string>();

  if (config.download_image) {
    const char* dir_path = "./results";
    struct stat info;
    if (stat(dir_path, &info) == 0 && S_ISDIR(info.st_mode)) {
      std::cout << "Directory already exists." << std::endl;
      int new_permissions = S_IRWXU | S_IRWXG | S_IRWXO;
      if (chmod(dir_path, new_permissions) == 0) {
        std::cout << "Directory permissions modified successfully."
                  << std::endl;
      } else {
        std::cerr << "Error modifying directory permissions." << std::endl;
        abort();
      }
    } else {
      if (mkdir(dir_path, 0777) == 0) {
        std::cout << "Directory created successfully." << std::endl;
      } else {
        std::cerr << "Error creating directory." << std::endl;
        abort();
      }
    }

    std::string line;
    if (demo_json.contains(JSON_CONFIG_CLASS_NAMES_FILED)) {
      istream.open(class_names_file);
      assert(istream.is_open());
      while (std::getline(istream, line)) {
        line = line.substr(0, line.length());
        config.class_names.push_back(line);
      }
      istream.close();
    }

    if (demo_json.contains(JSON_CONFIG_CAR_ATTRIBUTES_FILED)) {
      istream.open(car_attr_file);
      assert(istream.is_open());
      while (std::getline(istream, line)) {
        line = line.substr(0, line.length());
        config.car_attr.push_back(line);
      }
      istream.close();
    }
    if (demo_json.contains(JSON_CONFIG_PERSON_ATTRIBUTES_FILED)) {
      istream.open(person_attr_file);
      assert(istream.is_open());
      while (std::getline(istream, line)) {
        line = line.substr(0, line.length());
        config.person_attr.push_back(line);
      }
      istream.close();
    }
  }

  for (auto& channel_it : *channel_config_it) {
    nlohmann::json channel_json;
    channel_json["channel_id"] =
        channel_it.find(JSON_CONFIG_CHANNEL_CONFIG_CHANNEL_ID_FILED)
            ->get<int>();
    channel_json["url"] = channel_it.find(JSON_CONFIG_CHANNEL_CONFIG_URL_FILED)
                              ->get<std::string>();
    channel_json["source_type"] =
        channel_it.find(JSON_CONFIG_CHANNEL_CONFIG_SOURCE_TYPE_FILED)
            ->get<std::string>();

    channel_json["loop_num"] = 1;
    auto loop_num_it =
        channel_it.find(JSON_CONFIG_CHANNEL_CONFIG_LOOP_NUM_FILED);
    if (channel_it.end() != loop_num_it)
      channel_json["loop_num"] = loop_num_it->get<int>();

    auto fps_it = channel_it.find(JSON_CONFIG_CHANNEL_CONFIG_FPS_FILED);
    if (channel_it.end() != fps_it) channel_json["fps"] = fps_it->get<double>();

    auto roi_it = channel_it.find(JSON_CONFIG_CHANNEL_CONFIG_ROI_FILED);
    if (channel_it.end() != roi_it) channel_json["roi"] = *roi_it;

    auto sample_interval_it =
        channel_it.find(JSON_CONFIG_CHANNEL_CONFIG_SAMPLE_INTERVAL_FILED);
    if (channel_it.end() != sample_interval_it)
      channel_json["sample_interval"] = sample_interval_it->get<int>();

    auto sample_strategy_it =
        channel_it.find(JSON_CONFIG_CHANNEL_CONFIG_SAMPLE_STRATEGY_FILED);
    if (channel_it.end() != sample_strategy_it)
      channel_json["sample_strategy"] = sample_strategy_it->get<std::string>();

    auto skip_element_it =
        channel_it.find(JSON_CONFIG_CHANNEL_CONFIG_SKIP_ELEMENT_FILED);
    if (skip_element_it != channel_it.end()) {
      channel_json["skip_element"] = skip_element_it->get<std::vector<int>>();
    }

    channel_json["decode_id"] = -1;
    auto decode_idx_it = channel_it.find(JSON_CONFIG_CHANNEL_DECODE_IDX_FILED);
    if (decode_idx_it != channel_it.end()) {
      channel_json["decode_id"] = decode_idx_it->get<int>();
    }

    config.channel_configs.push_back(channel_json);
  }
  if (demo_json.contains(JSON_CONFIG_HTTP_REPORT_CONFIG_FILED)) {
    auto http_report_it = demo_json.find(JSON_CONFIG_HTTP_REPORT_CONFIG_FILED);
    config.report_config["port"] =
        http_report_it->find(JSON_CONFIG_HTTP_CONFIG_PORT_FILED)->get<int>();
    config.report_config["ip"] =
        http_report_it->find(JSON_CONFIG_HTTP_CONFIG_IP_FILED)
            ->get<std::string>();
    config.report_config["path"] =
        http_report_it->find(JSON_CONFIG_HTTP_CONFIG_PATH_FILED)
            ->get<std::string>();
  }
  if (demo_json.contains(JSON_CONFIG_HTTP_LISTEN_CONFIG_FILED)) {
    auto http_listen_it = demo_json.find(JSON_CONFIG_HTTP_LISTEN_CONFIG_FILED);
    config.listen_config["port"] =
        http_listen_it->find(JSON_CONFIG_HTTP_CONFIG_PORT_FILED)->get<int>();
    config.listen_config["ip"] =
        http_listen_it->find(JSON_CONFIG_HTTP_CONFIG_IP_FILED)
            ->get<std::string>();
    config.listen_config["path"] =
        http_listen_it->find(JSON_CONFIG_HTTP_CONFIG_PATH_FILED)
            ->get<std::string>();
  }
  return config;
}

bool bDispatch = false;
int switchtime_second = 10;
void* dispatchFun(void* data)
{
  IVS_INFO("ccy dispatchFun!");
  while(1)
  {
      // wait 
      sleep(switchtime_second);
      // change channel
      bDispatch = !bDispatch;
      IVS_INFO("ccy dispatchFun111! status:{0}",bDispatch);
  }
}


typedef struct MEMPACKED         //定义一个mem occupy的结构体  
{  
    char name1[20];      //定义一个char类型的数组名name有20个元素  
    unsigned long MemTotal;  
    char name2[20];  
    unsigned long MemFree;  
    char name3[20];  
    unsigned long Buffers;  
    char name4[20];  
    unsigned long Cached;  
    char name5[20];  
    unsigned long SwapCached;  
}MEM_OCCUPY;  

void get_memoccupy(MEM_OCCUPY *mem) //对无类型get函数含有一个形参结构体类弄的指针O  
{  
    FILE *fd;  
    char buff[256];  
    MEM_OCCUPY *m;  
    m = mem;  
      
    fd = fopen("/proc/meminfo", "r");  
    //MemTotal: 515164 kB  
    //MemFree: 7348 kB  
    //Buffers: 7892 kB  
    //Cached: 241852  kB  
    //SwapCached: 0 kB  
    //从fd文件中读取长度为buff的字符串再存到起始地址为buff这个空间里   
    fgets(buff, sizeof(buff), fd);  
    sscanf(buff, "%s %lu ", m->name1, &m->MemTotal);  
    fgets(buff, sizeof(buff), fd);  
    sscanf(buff, "%s %lu ", m->name2, &m->MemFree);  
    fgets(buff, sizeof(buff), fd);  
    sscanf(buff, "%s %lu ", m->name3, &m->Buffers);  
    fgets(buff, sizeof(buff), fd);  
    sscanf(buff, "%s %lu ", m->name4, &m->Cached);  
    fgets(buff, sizeof(buff), fd);   
    sscanf(buff, "%s %lu", m->name5, &m->SwapCached);  
      
    fclose(fd);     //关闭文件fd  
} 

int main(int argc, char* argv[]) {
  const char* keys =
      "{demo_config_path | "
      "../license_plate_recognition/config/license_plate_recognition_demo.json "
      "| demo config path}"
      "{help | 0 | print help information.}";
  cv::CommandLineParser parser(argc, argv, keys);
  if (parser.get<bool>("help")) {
    parser.printMessage();
    return 0;
  }
  std::string demo_config_fpath = parser.get<std::string>("demo_config_path");

  ::logInit("debug", "");

  std::mutex mtx;
  std::condition_variable cv;

  sophon_stream::common::Clocker clocker;
  std::atomic_uint32_t frameCount(0);
  std::atomic_int32_t finishedChannelCount(0);

  auto& engine = sophon_stream::framework::SingletonEngine::getInstance();

  std::ifstream istream;
  nlohmann::json engine_json;
  demo_config demo_json = parse_demo_json(demo_config_fpath);

  // 启动每个graph, graph之间没有联系，可以是完全不同的配置
  istream.open(demo_json.engine_config_file);
  STREAM_CHECK(istream.is_open(), "Please check if engine_config_file ",
               demo_json.engine_config_file, " exists.");
  istream >> engine_json;
  istream.close();

  demo_json.num_graphs = engine_json.size();
  demo_json.num_channels_per_graph = demo_json.channel_configs.size();
  int num_channels = demo_json.num_channels_per_graph * demo_json.num_graphs;

  // #if BMCV_VERSION_MAJOR > 1

  //   STREAM_CHECK(
  //       num_channels <= 1,
  //       "In order to ensure that the program can be run properly on the 1688,
  //       it " "is required that the number of input channels is less
  //       than 2.");

  // #endif

  std::vector<::sophon_stream::common::FpsProfiler> fpsProfilers(num_channels);
  for (int i = 0; i < num_channels; ++i) {
    std::string fpsName = "channel_" + std::to_string(i);
    fpsProfilers[i].config(fpsName, 100);
  }

  std::function<void(std::shared_ptr<sophon_stream::common::ObjectMetadata>)>
      draw_func;
  std::string out_dir = "./results";
  if (demo_json.draw_func_name == "draw_bytetrack_results")
    draw_func =
        std::bind(draw_bytetrack_results, std::placeholders::_1, out_dir);
  else if (demo_json.draw_func_name == "draw_license_plate_recognition_results")
    draw_func = std::bind(draw_license_plate_recognition_results,
                          std::placeholders::_1, out_dir);
  else if (demo_json.draw_func_name == "draw_openpose_results")
    draw_func =
        std::bind(draw_openpose_results, std::placeholders::_1, out_dir);
  else if (demo_json.draw_func_name == "draw_retinaface_results")
    draw_func =
        std::bind(draw_retinaface_results, std::placeholders::_1, out_dir);
  else if (demo_json.draw_func_name ==
           "draw_retinaface_distributor_resnet_faiss_converger_results")
    draw_func =
        std::bind(draw_retinaface_distributor_resnet_faiss_converger_results,
                  std::placeholders::_1, out_dir);
  else if (demo_json.draw_func_name == "draw_yolov5_results")
    draw_func = std::bind(draw_yolov5_results, std::placeholders::_1, out_dir,
                          demo_json.class_names);
  else if (demo_json.draw_func_name ==
           "draw_yolov5_bytetrack_distributor_resnet_converger_results")
    draw_func =
        std::bind(draw_yolov5_bytetrack_distributor_resnet_converger_results,
                  std::placeholders::_1, out_dir, demo_json.car_attr,
                  demo_json.person_attr);
  else if (demo_json.draw_func_name == "draw_yolox_results")
    draw_func = std::bind(draw_yolox_results, std::placeholders::_1, out_dir,
                          demo_json.class_names);
  else if (demo_json.draw_func_name == "draw_yolov7_results")
    draw_func = std::bind(draw_yolov5_results, std::placeholders::_1, out_dir,
                          demo_json.class_names);
  else if (demo_json.draw_func_name == "save_only")
    draw_func = std::bind(save_only, std::placeholders::_1, out_dir);
  else if (demo_json.draw_func_name == "draw_yolov5_fastpose_posec3d_results")
    draw_func =
        std::bind(draw_yolov5_fastpose_posec3d_results, std::placeholders::_1,
                  out_dir, demo_json.heatmap_loss);
  else if (demo_json.draw_func_name == "default")
    draw_func = std::function<void(
        std::shared_ptr<sophon_stream::common::ObjectMetadata>)>(draw_default);
  else if (demo_json.draw_func_name == "draw_ppocr_results")
    draw_func = std::bind(draw_ppocr_results, std::placeholders::_1, out_dir);
  else if (demo_json.draw_func_name == "draw_yolov8_det_pose")
    draw_func = std::bind(draw_yolov8_det_pose, std::placeholders::_1, out_dir);
  else
    IVS_ERROR("No such function! Please check your 'draw_func_name'.");

  auto sinkHandler = [&, draw_func](std::shared_ptr<void> data) {
    // write stop data handler here
    auto objectMetadata =
        std::static_pointer_cast<sophon_stream::common::ObjectMetadata>(data);
    if (objectMetadata == nullptr) return;
    if (!objectMetadata->mFilter) {
      frameCount++;
      fpsProfilers[objectMetadata->mFrame->mChannelIdInternal].add(1);
    }
    if (objectMetadata->mFrame->mEndOfStream) {
      printf("meet a eof\n");
      finishedChannelCount++;
      if (finishedChannelCount == num_channels) {
        cv.notify_one();
      }
      return;
    }
    if (demo_json.download_image) draw_func(objectMetadata);
  };
  sophon_stream::framework::ListenThread* listenthread =
      sophon_stream::framework::ListenThread::getInstance();
  listenthread->init(demo_json.report_config, demo_json.listen_config);
  engine.setListener(listenthread);
  std::map<int, std::vector<std::pair<int, int>>> graph_src_id_port_map;
  init_engine(engine, engine_json, sinkHandler, graph_src_id_port_map);
  IVS_INFO("ccy engine!");

  for (auto graph_id : engine.getGraphIds()) {
    for (auto& channel_config : demo_json.channel_configs) {
      auto channelTask =
          std::make_shared<sophon_stream::element::decode::ChannelTask>();
      channelTask->request.operation = sophon_stream::element::decode::
          ChannelOperateRequest::ChannelOperate::START;
      channelTask->request.channelId = channel_config["channel_id"];
      channelTask->request.json = channel_config.dump();
      int decode_id = channel_config["decode_id"];
      // std::pair<int, int> src_id_port =
      // graph_src_id_port_map[graph_id][decode_id];

      auto src_id_port_vec = graph_src_id_port_map[graph_id];
      for (auto& src_id_port : src_id_port_vec) {
        // decode_id == -1为默认情况，即只有一个解码器
        // decode_id != -1，即有多个解码器，要求每个都写清参数
        if ((decode_id == -1 && src_id_port_vec.size() == 1) || src_id_port.first == decode_id) {
          sophon_stream::common::ErrorCode errorCode = engine.pushSourceData(
              graph_id, src_id_port.first, src_id_port.second,
              std::static_pointer_cast<void>(channelTask));
        } else {
          IVS_ERROR("Push Source Data Failed! Please Check Input Json!");
        }
      }
    }
  }
  IVS_INFO("ccy startEnd!");

  // add for test ccy
  pthread_t dispatch_thread;

  if(!demo_json.cameras_config_file.empty())
  {
    pthread_create(&dispatch_thread, NULL, dispatchFun, NULL); 

    camera_list_config cameralistConfig;
    parse_camera_list_json(demo_json.cameras_config_file, cameralistConfig);
    camera_sorted_info sortedInfo;
    sort_camera_list_info(cameralistConfig, sortedInfo);

    IVS_INFO("ccy cameralist! channelCount:{0}",sortedInfo.channels.size());

    bool bRun = true;
    bool lastStatus = false;
    int listIndex = 0;
    MEM_OCCUPY mem_stat;

    while(bRun)
    {
      if(bDispatch != lastStatus)
      {
        lastStatus = bDispatch;
        IVS_INFO("ccy dispatchStart! status:{0}",lastStatus);
        
        for (auto graph_id : engine.getGraphIds()) {
          for (auto& channel_config : demo_json.channel_configs) {

            for(int i = 0; i < sortedInfo.channels.size(); i ++)
            {
              if(channel_config["channel_id"] == sortedInfo.channels.at(i).channel_id)
              {
                auto channelTask =
                  std::make_shared<sophon_stream::element::decode::ChannelTask>();

                channelTask->request.operation = sophon_stream::element::decode::
                    ChannelOperateRequest::ChannelOperate::STOP;
                channelTask->request.channelId = channel_config["channel_id"];
                channelTask->request.json = channel_config.dump();
                int decode_id = channel_config["decode_id"];

                auto src_id_port_vec = graph_src_id_port_map[graph_id];
                for (auto& src_id_port : src_id_port_vec) {
                  IVS_INFO("ccy dispatch! decodeid:{0},portSize:{1}",decode_id,src_id_port_vec.size());
                  // decode_id == -1为默认情况，即只有一个解码器
                  // decode_id != -1，即有多个解码器，要求每个都写清参数
                  if ((decode_id == -1 && src_id_port_vec.size() == 1) || src_id_port.first == decode_id) {
                    sophon_stream::common::ErrorCode errorCode = engine.pushSourceData(
                        graph_id, src_id_port.first, src_id_port.second,
                        std::static_pointer_cast<void>(channelTask));
                  } else {
                    IVS_ERROR("Push Source Data Failed! Please Check Input Json!");
                  }
                }

              }
            }
          }
        }

        IVS_INFO("ccy dispatch stop!");
        get_memoccupy((MEM_OCCUPY *)&mem_stat);
        cout << "TotalMem:" << mem_stat.MemTotal
            << "Free:" << mem_stat.MemFree
            << "Buffer:" << mem_stat.Buffers
            << endl;

        for (auto graph_id : engine.getGraphIds()) {
          for (auto& channel_config : demo_json.channel_configs) {
            for(int i = 0; i < sortedInfo.channels.size(); i ++)
            {
              if(channel_config["channel_id"] == sortedInfo.channels.at(i).channel_id)
              {
                auto channelTask =
                  std::make_shared<sophon_stream::element::decode::ChannelTask>();

                channelTask->request.operation = sophon_stream::element::decode::
                    ChannelOperateRequest::ChannelOperate::START;
                channelTask->request.channelId = channel_config["channel_id"];

                int index = sortedInfo.channels.at(i).current_url_index;
                IVS_INFO("ccy dispatching channel = {0}, urlIndex = {1}", sortedInfo.channels.at(i).channel_id, index);
                
                channel_config["url"] = sortedInfo.channels.at(i).urls.at(index);
                index ++;
                if(index >= sortedInfo.channels.at(i).urls.size())
                {
                  index = 0;
                }
                sortedInfo.channels.at(i).current_url_index = index;

                channelTask->request.json = channel_config.dump();
                int decode_id = channel_config["decode_id"];

                auto src_id_port_vec = graph_src_id_port_map[graph_id];
                for (auto& src_id_port : src_id_port_vec) {
                  // decode_id == -1为默认情况，即只有一个解码器
                  // decode_id != -1，即有多个解码器，要求每个都写清参数
                  if ((decode_id == -1 && src_id_port_vec.size() == 1) || src_id_port.first == decode_id) {
                    sophon_stream::common::ErrorCode errorCode = engine.pushSourceData(
                        graph_id, src_id_port.first, src_id_port.second,
                        std::static_pointer_cast<void>(channelTask));
                  } else {
                    IVS_ERROR("Push Source Data Failed! Please Check Input Json!");
                  }
                }

              }
            }
            
          }
        }
      }
    }
    get_memoccupy((MEM_OCCUPY *)&mem_stat);
    IVS_INFO("ccy addEnd!");
  }
  // add end

  {
    std::unique_lock<std::mutex> uq(mtx);
    cv.wait(uq);
    std::cout << "ccy wait" << std::endl;
  }
  for (int i = 0; i < demo_json.num_graphs; i++) {
    std::cout << "graph stop" << std::endl;
    engine.stop(i);
  }
  long totalCost = clocker.tell_us();
  std::cout << " total time cost " << totalCost << " us." << std::endl;
  double fps = static_cast<double>(frameCount) / totalCost;
  std::cout << "frame count is " << frameCount << " | fps is " << fps * 1000000
            << " fps." << std::endl;


  pthread_join(dispatch_thread, NULL);

  return 0;
}