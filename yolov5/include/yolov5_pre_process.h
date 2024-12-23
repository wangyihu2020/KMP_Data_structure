//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-STREAM is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef SOPHON_STREAM_ELEMENT_YOLOV5_PRE_PROCESS_H_
#define SOPHON_STREAM_ELEMENT_YOLOV5_PRE_PROCESS_H_

#include "algorithmApi/pre_process.h"
#include "yolov5_context.h"

namespace nvr_edge {
namespace backend {
namespace stream_elements {
namespace yolov5 {

class Yolov5PreProcess : public stream_elements::PreProcess {
 public:
  common::nvr_error_code_c preProcess(std::shared_ptr<Yolov5Context> context,
                               stream::object_meta_datas& object_meta_datas);
  void init(std::shared_ptr<Yolov5Context> context);
};

}  // namespace yolov5
}  //namespace stream_elements
}  //namespace backend
}  //namespace nvr_edge

#endif  // SOPHON_STREAM_ELEMENT_YOLOV5_PRE_PROCESS_H_