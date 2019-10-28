#include <torch/torch.h>
#include <torch/extension.h>
#include <math.h>
#include <iostream>
#include "roi_align_kernel.h"

using namespace std;

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
    return 1;
  }
  // Number of ROIs
  int num_rois = rois.size(0);

  // height
  int height = features.size(2);
  // width
  int width = features.size(3);
  // Number of channels
  int channels = features.size(1);

  cout << "roi_align_forward_cuda" << endl;
  ROIAlignForwardLaucher(spatial_scale,
                         aligned_height,
                         aligned_width, num_rois,
                         height,
                         width,
                         channels,
                         features.data_ptr<float>(),
                         rois.data_ptr<float>(),
                         output.data_ptr<float>());
  return 0;

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
    return 1;
  }

  // Number of ROIs
  int num_rois = rois.size(0);

  // height
  int height = bottom_grad.size(2);
  // width
  int width = bottom_grad.size(3);
  // Number of channels
  int channels = bottom_grad.size(1);

  cout << "roi_align_forward_cuda" << endl;
  ROIAlignBackwardLaucher(spatial_scale,
                          aligned_height,
                          aligned_width, num_rois,
                          height,
                          width,
                          channels,
                          top_grad.data_ptr<float>(),
                          rois.data_ptr<float>(),
                          bottom_grad.data_ptr<float>());
  return 0;

}


