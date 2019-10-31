#include <torch/torch.h>
#include <THC/THC.h>
#include <stdio.h>
#include "nms_cuda_kernel.h"

// this symbol will be resolved automatically from PyTorch ..
extern THCState *state;

int nms_cuda(torch::Tensor keep_out,
             torch::Tensor boxes_host,
             torch::Tensor num_out,
             float nms_overlap_thresh) {

  nms_cuda_compute(keep_out.data_ptr<int>(),
                   num_out.data_ptr<int>(),
                   boxes_host.data_ptr<float>(),
                   boxes_host.size(0),
                   boxes_host.size(1),
                   nms_overlap_thresh);

  return 1;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms_cuda", &nms_cuda, "nms cuda");
}