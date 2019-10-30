#include <torch/torch.h>
#include <math.h>
#include "roi_pooling_kernel.h"


int roi_pooling_forward_cuda(int pooled_height,
                             int pooled_width,
                             float spatial_scale,
                             torch::Tensor features,
                             torch::Tensor rois,
                             torch::Tensor output,
                             torch::Tensor argmax)
{
  // Grab the input tensor
  float * data_flat = features.data_ptr<float>();
  float * rois_flat = rois.data_ptr<float>();

  float * output_flat = output.data_ptr<float>();
  int * argmax_flat = argmax.data_ptr<int>();

  // Number of ROIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);
  if (size_rois != 5)
  {
    return 1;
  }


  // data height
  int data_height = features.size(2);
  // data width
  int data_width = features.size(3);
  // Number of channels
  int num_channels = features.size(1);


  ROIPoolForwardLaucher(data_flat,
                        spatial_scale,
                        num_rois,
                        data_height,
                        data_width,
                        num_channels,
                        pooled_height,
                        pooled_width,
                        rois_flat,
                        output_flat,
                        argmax_flat);

  return 0;
}

int roi_pooling_backward_cuda(int pooled_height,
                              int pooled_width,
                              float spatial_scale,
                              torch::Tensor top_grad,
                              torch::Tensor rois,
                              torch::Tensor bottom_grad,
                              torch::Tensor argmax)
{
  // Grab the input tensor
  float * top_grad_flat = top_grad.data_ptr<float>();
  float * rois_flat = rois.data_ptr<float>();

  float * bottom_grad_flat = bottom_grad.data_ptr<float>();
  int * argmax_flat = argmax.data_ptr<int>();

  // Number of ROIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);
  if (size_rois != 5)
  {
    return 1;
  }

  // batch size
  int batch_size = bottom_grad.size(0);

  int data_height = bottom_grad.size(2);
  // data width
  int data_width = bottom_grad.size(3);
  // Number of channels
  int num_channels = bottom_grad.size(1);

  ROIPoolBackwardLaucher(top_grad_flat,
                         spatial_scale,
                         batch_size,
                         num_rois,
                         data_height,
                         data_width,
                         num_channels,
                         pooled_height,
                         pooled_width,
                         rois_flat,
                         bottom_grad_flat,
                         argmax_flat);

  return 0;
}
