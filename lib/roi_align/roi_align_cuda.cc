#include <torch/torch.h>
#include <torch/extension.h>
#include <THC/THC.h>
#include <math.h>
#include "roi_align_kernel.h"


extern THCState *state;

int roi_align_forward_cuda(int aligned_height,
                           int aligned_width,
                           float spatial_scale,
                           torch::Tensor features,
                           torch::Tensor rois,
                           torch::Tensor output)
{
  int size_rois = rois.size(1);
  if (size_rois != 5)
  {
    return 0;
  }

  cudaStream_t stream = THCState_getCurrentStream(state);

  ROIAlignForwardLaucher(features, spatial_scale, aligned_height,
                         aligned_width, rois, output, stream);

  return 1;
}

int roi_align_backward_cuda(int aligned_height,
                            int aligned_width,
                            float spatial_scale,
                            torch::Tensor top_grad,
                            torch::Tensor rois,
                            torch::Tensor bottom_grad)
{

  // Number of ROIs
  int size_rois = rois.size(1);
  if (size_rois != 5)
  {
    return 0;
  }

  cudaStream_t stream = THCState_getCurrentStream(state);
  ROIAlignBackwardLaucher(top_grad, spatial_scale, aligned_height,
                          aligned_width, rois, bottom_grad, stream);

  return 1;
}


