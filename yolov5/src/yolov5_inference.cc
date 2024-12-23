//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-STREAM is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "yolov5_inference.h"

namespace nvr_edge {
namespace backend {
namespace stream_elements {
namespace yolov5 {

Yolov5Inference::~Yolov5Inference() {}

void Yolov5Inference::init(std::shared_ptr<Yolov5Context> context) {}

common::nvr_error_code_c Yolov5Inference::predict(
    std::shared_ptr<Yolov5Context> context,
    stream::object_meta_datas& object_meta_datas) {
  if (object_meta_datas.size() == 0) return common::nvr_error_code_c::SUCCESS;

  if (context->max_batch > 1) {
    auto inputTensors = mergeInputDeviceMem(context, object_meta_datas);
    auto outputTensors = getOutputDeviceMem(context);

    int ret = 0;
#if BMCV_VERSION_MAJOR > 1
    ret = context->bmNetwork->forward<false>(inputTensors->tensors,
                                             outputTensors->tensors);
#else
    ret = context->bmNetwork->forward(inputTensors->tensors,
                                      outputTensors->tensors);
#endif

    splitOutputMemIntoobject_meta_datas(context, object_meta_datas, outputTensors);
  } else {
    if (object_meta_datas[0]->m_frame->mEndOfStream)
      return common::nvr_error_code_c::SUCCESS;
    object_meta_datas[0]->mOutputBMtensors = getOutputDeviceMem(context);
#if BMCV_VERSION_MAJOR > 1
    int ret = context->bmNetwork->forward<false>(
        object_meta_datas[0]->mInputBMtensors->tensors,
        object_meta_datas[0]->mOutputBMtensors->tensors);
#else
    int ret = context->bmNetwork->forward(
        object_meta_datas[0]->mInputBMtensors->tensors,
        object_meta_datas[0]->mOutputBMtensors->tensors);
#endif
  }

  for (auto obj : object_meta_datas) {
    obj->mInputBMtensors = nullptr;
  }

  return common::nvr_error_code_c::SUCCESS;
}

}  // namespace yolov5
}  //namespace stream_elements
}  //namespace backend
}  //namespace nvr_edge