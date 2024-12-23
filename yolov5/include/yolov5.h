//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-STREAM is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef SOPHON_STREAM_ELEMENT_YOLOV5_H_
#define SOPHON_STREAM_ELEMENT_YOLOV5_H_

#include "element_factory.h"
#include "group.h"
#include "yolov5_context.h"
#include "yolov5_inference.h"
#include "yolov5_post_process.h"
#include "yolov5_pre_process.h"
#include <httplib.h>
#include <profiler.h>

namespace nvr_edge {
namespace backend {
namespace stream_elements {
namespace yolov5 {

/**
 * 算法模块
 */
class Yolov5 : public stream::element_c {
 public:
  Yolov5(const char* name);
  ~Yolov5() override;

  const static std::string elementName;

  common::nvr_error_code_c init_internal(const std::string& json) override;

  common::nvr_error_code_c element_do_work(int data_pipe_id) override;

  void setContext(std::shared_ptr<stream_elements::Context> context);
  void setPreprocess(std::shared_ptr<stream_elements::PreProcess> pre);
  void setInference(std::shared_ptr<stream_elements::Inference> infer);
  void setPostprocess(
      std::shared_ptr<stream_elements::PostProcess> post);
  void setStage(bool pre, bool infer, bool post);
  void initProfiler(std::string name, int interval);
  std::shared_ptr<stream_elements::Context> getContext() {
    return mContext;
  }
  std::shared_ptr<stream_elements::PreProcess> getPreProcess() {
    return mPreProcess;
  }
  std::shared_ptr<stream_elements::Inference> getInference() {
    return mInference;
  }
  std::shared_ptr<stream_elements::PostProcess> getPostProcess() {
    return mPostProcess;
  }

  std::string postNameSetConfThreshold = "/yolov5/SetConfThreshold";
  void listenerSetConfThreshold(const httplib::Request& request,
                                httplib::Response& response);

  static constexpr const char* CONFIG_INTERNAL_STAGE_NAME_FIELD = "stage";
  static constexpr const char* CONFIG_INTERNAL_MODEL_PATH_FIELD = "model_path";
  static constexpr const char* CONFIG_INTERNAL_THRESHOLD_CONF_FIELD =
      "threshold_conf";
  static constexpr const char* CONFIG_INTERNAL_THRESHOLD_NMS_FIELD =
      "threshold_nms";
  static constexpr const char* CONFIG_INTERNAL_THRESHOLD_TPU_KERNEL_FIELD =
      "use_tpu_kernel";
  static constexpr const char* CONFIG_INTERNAL_THRESHOLD_BGR2RGB_FIELD =
      "bgr2rgb";
  static constexpr const char* CONFIG_INTERNAL_THRESHOLD_MEAN_FIELD = "mean";
  static constexpr const char* CONFIG_INTERNAL_THRESHOLD_STD_FIELD = "std";
  static constexpr const char* CONFIG_INTERNAL_CLASS_NAMES_FILE_FIELD =
      "class_names_file";
  static constexpr const char* CONFIG_INTERNAL_ROI_FILED = "roi";
  static constexpr const char* CONFIG_INTERNAL_LEFT_FILED = "left";
  static constexpr const char* CONFIG_INTERNAL_TOP_FILED = "top";
  static constexpr const char* CONFIG_INTERNAL_WIDTH_FILED = "width";
  static constexpr const char* CONFIG_INTERNAL_HEIGHT_FILED = "height";
  static constexpr const char* CONFIG_INTERNAL_MAX_DET_FILED = "maxdet";
  static constexpr const char* CONFIG_INTERNAL_MIN_DET_FILED = "mindet";

 private:
  std::shared_ptr<Yolov5Context> mContext;          // context对象
  std::shared_ptr<Yolov5PreProcess> mPreProcess;    // 预处理对象
  std::shared_ptr<Yolov5Inference> mInference;      // 推理对象
  std::shared_ptr<Yolov5PostProcess> mPostProcess;  // 后处理对象

  bool use_pre = false;
  bool use_infer = false;
  bool use_post = false;

  std::string mFpsProfilerName;
  stream::FpsProfiler mFpsProfiler;

  common::nvr_error_code_c initContext(const std::string& json);
  void process(stream::object_meta_datas& object_meta_datas, int data_pipe_id);
};

}  // namespace yolov5
}  //namespace stream_elements
}  //namespace backend
}  //namespace nvr_edge

#endif  // SOPHON_STREAM_ELEMENT_YOLOV5_H_