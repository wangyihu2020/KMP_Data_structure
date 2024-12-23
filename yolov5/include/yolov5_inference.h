//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-STREAM is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef SOPHON_STREAM_ELEMENT_YOLOV5_INFERENCE_H_
#define SOPHON_STREAM_ELEMENT_YOLOV5_INFERENCE_H_

#include "algorithmApi/inference.h"
#include "yolov5_context.h"

namespace nvr_edge {
namespace backend {
namespace stream_elements {
namespace yolov5 {

class Yolov5Inference : public stream_elements::Inference {
 public:
  ~Yolov5Inference() override;
  /**
   * init device and engine
   * @param[in] context: model path,inputs and outputs name...
   */
  void init(std::shared_ptr<Yolov5Context> context);

  /**
   * network predict output
   * @param[in] context: inputData and outputData
   */
  common::nvr_error_code_c predict(std::shared_ptr<Yolov5Context> context,
                            stream::object_meta_datas& object_meta_datas);

 private:
};

}  // namespace yolov5
}  //namespace stream_elements
}  //namespace backend
}  //namespace nvr_edge

#endif  // SOPHON_STREAM_ELEMENT_YOLOV5_INFERENCE_H_