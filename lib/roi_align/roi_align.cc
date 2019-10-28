#include <torch/torch.h>
#include <math.h>
#include <omp.h>
#include <iostream>
#include <string>
#include "roi_align_cuda.h"

using namespace std;


void ROIAlignForwardCpu(const float spatial_scale,
                        const int aligned_height,
                        const int aligned_width,
                        torch::Tensor features,
                        torch::Tensor rois,
                        torch::Tensor output)

{
  // Number of ROIs
  int num_rois = rois.size(0);

  // height
  int height = features.size(2);
  // width
  int width = features.size(3);
  // Number of channels
  int channels = features.size(1);


  for (int n = 0; n < num_rois; ++n)
  {
    int roi_bi = (int)rois[n][0].item<float>();
    float roi_start_w = rois[n][1].item<float>() * spatial_scale;
    float roi_start_h = rois[n][2].item<float>() * spatial_scale;
    float roi_end_w = rois[n][3].item<float>() * spatial_scale;
    float roi_end_h = rois[n][4].item<float>() * spatial_scale;

    // Force malformed ROI to be 1x1
    float roi_width = fmaxf(roi_end_w - roi_start_w + 1., 0.);
    float roi_height = fmaxf(roi_end_h - roi_start_h + 1., 0.);
    float bin_size_h = roi_height / (aligned_height - 1.);
    float bin_size_w = roi_width / (aligned_width - 1.);

    for (int ph = 0; ph < aligned_height; ++ph)
    {
      float h = (float)(ph) * bin_size_h + roi_start_h;
      int hs = fminf(floor(h), height - 2);
      float h_ratio = h - (float)(hs);

      for (int pw = 0; pw < aligned_width; ++pw)
      {
        float w = (float)(pw) * bin_size_w + roi_start_w;
        int ws = fminf(floor(w), width - 2);
        float w_ratio = w - (float)(ws);

        // bilinear interpolation
        if (h < 0 || h >= height || w < 0 || w >= width)
        {
          for (int c = 0; c < channels; ++c)
          {
            output[n][c][h][w] = 0.;
          }

        } else {

          for (int c = 0; c < channels; ++c)
          {
            output[n][c][ph][pw] = 0.;
            output[n][c][ph][pw] += features[roi_bi][c][hs][ws] * (1. - h_ratio) * (1. - w_ratio);
            output[n][c][ph][pw] += features[roi_bi][c][hs][ws + 1] * (1. - h_ratio) * w_ratio;
            output[n][c][ph][pw] += features[roi_bi][c][hs + 1][ws] * h_ratio * (1. - w_ratio);
            output[n][c][ph][pw] += features[roi_bi][c][hs + 1][ws + 1] * h_ratio * w_ratio;

          }
        }
      }
    }
  }
}

void ROIAlignBackwardCpu(const float spatial_scale,
                         const int aligned_height,
                         const int aligned_width,
                         torch::Tensor top_grad,
                         torch::Tensor rois,
                         torch::Tensor bottom_grad
                        )
{

  // Number of ROIs
  int num_rois = rois.size(0);

  // batch size
  // int batch_size = bottom_grad.size(0);
  // Number of channels
  int channels = bottom_grad.size(1);
  // data height
  int height = bottom_grad.size(2);
  // data width
  int width = bottom_grad.size(3);

  for (int n = 0; n < num_rois; ++n)
  {
    int roi_bi = (int)rois[n][0].item<float>();
    float roi_start_w = rois[n][1].item<float>() * spatial_scale;
    float roi_start_h = rois[n][2].item<float>() * spatial_scale;
    float roi_end_w = rois[n][3].item<float>() * spatial_scale;
    float roi_end_h = rois[n][4].item<float>() * spatial_scale;

    // Force malformed ROI to be 1x1
    float roi_width = fmaxf(roi_end_w - roi_start_w + 1., 0.);
    float roi_height = fmaxf(roi_end_h - roi_start_h + 1., 0.);
    float bin_size_h = roi_height / (aligned_height - 1.);
    float bin_size_w = roi_width / (aligned_width - 1.);


    for (int ph = 0; ph < aligned_height; ++ph)
    {

      float h = (float)(ph) * bin_size_h + roi_start_h;
      int hs = fminf(floor(h), height - 2);
      for (int pw = 0; pw < aligned_width; ++pw)
      {

        float w = (float)(pw) * bin_size_w + roi_start_w;
        int ws = fminf(floor(w), width - 2);

        // bilinear interpolation
        if (h >= 0 && h < height && w >= 0 && w < width)
        {
          float h_ratio = h - (float)(hs);
          float w_ratio = w - (float)(ws);
          for (int c = 0; c < channels; ++c)
          {

            bottom_grad[roi_bi][c][hs][ws] += top_grad[n][c][ph][pw] * (1. - h_ratio) * (1. - w_ratio);
            bottom_grad[roi_bi][c][hs][ws + 1] += top_grad[n][c][ph][pw] * (1. - h_ratio) *  w_ratio;
            bottom_grad[roi_bi][c][hs + 1][ws] += top_grad[n][c][ph][pw] * h_ratio * (1. - w_ratio);
            bottom_grad[roi_bi][c][hs + 1][ws + 1] += top_grad[n][c][ph][pw] * h_ratio * w_ratio;

          }
        }
      }
    }
  }
}

int roi_align_forward(int aligned_height,
                      int aligned_width,
                      float spatial_scale,
                      torch::Tensor features,
                      torch::Tensor rois,
                      torch::Tensor output)
{

  // Number of ROIs
  int size_rois = rois.size(1);
  if (size_rois != 5)
  {
    return 0;
  }
  ROIAlignForwardCpu(spatial_scale,
                     aligned_height,
                     aligned_width,
                     features,
                     rois,
                     output);

  return 1;
}

int roi_align_backward(int aligned_height,
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
  ROIAlignBackwardCpu(spatial_scale,
                      aligned_height,
                      aligned_width,
                      top_grad,
                      rois,
                      bottom_grad);

  return 1;
}


int roi_align_forward_gpu(int aligned_height,
                          int aligned_width,
                          float spatial_scale,
                          torch::Tensor features,
                          torch::Tensor rois,
                          torch::Tensor output) {

  return roi_align_forward_cuda(aligned_height,
                                aligned_width,
                                spatial_scale,
                                features,
                                rois, output);
}

int roi_align_backward_gpu(int aligned_height,
                           int aligned_width,
                           float spatial_scale,
                           torch::Tensor top_grad,
                           torch::Tensor rois,
                           torch::Tensor bottom_grad) {

  return roi_align_backward_cuda(aligned_height,
                                 aligned_width,
                                 spatial_scale,
                                 top_grad,
                                 rois, bottom_grad);
}


torch::Tensor roi_align_cuda_test(torch::Tensor a, torch::Tensor b) {
  torch::Tensor res = torch::zeros_like(a, torch::device(torch::kCUDA).dtype(torch::kFloat));
  // res = a + b;
  cout << "a.dtype:" << a.dtype() << endl;
  cout << "b.dtype:" << b.dtype() << endl;
  const auto rows = a.size(0);
  const auto cols = a.size(1);
  char buff[100];
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {

      snprintf(buff, sizeof(buff),
               "a[%d][%d]:%f,b[%d][%d]:%f",
               i, j, a[i][j].item<float>(),
               i, j, b[i][j].item<float>());
      std::string buffStr = buff;
      cout << buffStr << endl;
      res[i][j] = 7.3;
    }
  }
  return res;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("roi_align_forward", &roi_align_forward, "roi_align_forward");
  m.def("roi_align_backward", &roi_align_backward, "roi_align_backward");
  m.def("roi_align_forward_gpu", &roi_align_forward_gpu, "roi_align_forward_gpu");
  m.def("roi_align_backward_gpu", &roi_align_backward_gpu, "roi_align_backward_gpu");
  m.def("roi_align_cuda_test", &roi_align_cuda_test, "roi_align_cuda_test");
}

