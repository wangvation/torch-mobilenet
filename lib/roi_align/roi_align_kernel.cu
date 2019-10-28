#ifdef __cplusplus
extern "C" {
#endif
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "roi_align_kernel.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
            i += blockDim.x * gridDim.x)


__global__ void ROIAlignForward(const int nthreads,
                                const torch::Tensor bottom_data,
                                const float spatial_scale,
                                const int height,
                                const int width,
                                const int channels,
                                const int aligned_height,
                                const int aligned_width,
                                const torch::Tensor bottom_rois,
                                torch::Tensor top_data) {

  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       i < nthreads;
       index += blockDim.x * gridDim.x) {
    // (n, c, ph, pw) is an element in the aligned output
    // int n = index;
    // int pw = n % aligned_width;
    // n /= aligned_width;
    // int ph = n % aligned_height;
    // n /= aligned_height;
    // int c = n % channels;
    // n /= channels;

    int pw = index % aligned_width;
    int ph = (index / aligned_width) % aligned_height;
    int c  = (index / aligned_width / aligned_height) % channels;
    int n  = index / aligned_width / aligned_height / channels;

    // bottom_rois += n * 5;
    int roi_bi = floor(bottom_rois[n][0]);
    float roi_start_w = bottom_rois[n][1] * spatial_scale;
    float roi_start_h = bottom_rois[n][2] * spatial_scale;
    float roi_end_w = bottom_rois[n][3] * spatial_scale;
    float roi_end_h = bottom_rois[n][4] * spatial_scale;

    // Force malformed ROIs to be 1x1
    float roi_width = fmaxf(roi_end_w - roi_start_w + 1., 0.);
    float roi_height = fmaxf(roi_end_h - roi_start_h + 1., 0.);
    float bin_size_h = roi_height / (aligned_height - 1.);
    float bin_size_w = roi_width / (aligned_width - 1.);

    float h = (float)(ph) * bin_size_h + roi_start_h;
    float w = (float)(pw) * bin_size_w + roi_start_w;

    int hs = fminf(floor(h), height - 2);
    int ws = fminf(floor(w), width - 2);


    // bilinear interpolation
    if (h < 0 || h >= height || w < 0 || w >= width) {
      top_data[index] = 0.;
    } else {
      float h_ratio = h - (float)(hs);
      float w_ratio = w - (float)(ws);

      output[n][c][ph][pw] = 0.;
      output[n][c][ph][pw] += bottom_data[roi_bi][c][hs][ws] * (1. - h_ratio) * (1. - w_ratio);
      output[n][c][ph][pw] += bottom_data[roi_bi][c][hs][ws + 1] * (1. - h_ratio) * w_ratio;
      output[n][c][ph][pw] += bottom_data[roi_bi][c][hs + 1][ws] * h_ratio * (1. - w_ratio);
      output[n][c][ph][pw] += bottom_data[roi_bi][c][hs + 1][ws + 1] * h_ratio * w_ratio;
    }
  }
}


int ROIAlignForwardLaucher(const torch::Tensor features,
                           const float spatial_scale,
                           const int aligned_height,
                           const int aligned_width,
                           const torch::Tensor rois,
                           torch::Tensor output,
                           cudaStream_t stream) {
  const int kThreadsPerBlock = 1024;
  // Number of ROIs
  int num_rois = rois.size(0);
  // data height
  int height = features.size(2);
  // data width
  int width = features.size(3);
  // Number of channels
  int channels = features.size(1);
  const int output_size = num_rois * aligned_height * aligned_width * channels;
  cudaError_t err;


  ROIAlignForward <<< (output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream >>> (
    output_size, features, spatial_scale, height, width, channels,
    aligned_height, aligned_width, rois, output);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return 1;
}


__global__ void ROIAlignBackward(const int nthreads,
                                 const torch::Tensor top_diff,
                                 const float spatial_scale,
                                 const int height,
                                 const int width,
                                 const int channels,
                                 const int aligned_height,
                                 const int aligned_width,
                                 torch::Tensor bottom_diff,
                                 const torch::Tensor bottom_rois) {

  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       i < nthreads;
       index += blockDim.x * gridDim.x) {

    // (n, c, ph, pw) is an element in the aligned output
    int pw = index % aligned_width;
    int ph = (index / aligned_width) % aligned_height;
    int c  = (index / aligned_width / aligned_height) % channels;
    int n  = index / aligned_width / aligned_height / channels;

    int roi_bi = (int)bottom_rois[n][0];
    float roi_start_w = bottom_rois[n][1] * spatial_scale;
    float roi_start_h = bottom_rois[n][2] * spatial_scale;
    float roi_end_w = bottom_rois[n][3] * spatial_scale;
    float roi_end_h = bottom_rois[n][4] * spatial_scale;
    /* int roi_start_w = round(bottom_rois[1] * spatial_scale); */
    /* int roi_start_h = round(bottom_rois[2] * spatial_scale); */
    /* int roi_end_w = round(bottom_rois[3] * spatial_scale); */
    /* int roi_end_h = round(bottom_rois[4] * spatial_scale); */

    // Force malformed ROIs to be 1x1
    float roi_width = fmaxf(roi_end_w - roi_start_w + 1., 0.);
    float roi_height = fmaxf(roi_end_h - roi_start_h + 1., 0.);
    float bin_size_h = roi_height / (aligned_height - 1.);
    float bin_size_w = roi_width / (aligned_width - 1.);

    float h = (float)(ph) * bin_size_h + roi_start_h;
    float w = (float)(pw) * bin_size_w + roi_start_w;

    int hs = fminf(floor(h), height - 2);
    int ws = fminf(floor(w), width - 2);


    // bilinear interpolation
    if (!(h < 0 || h >= height || w < 0 || w >= width)) {
      float h_ratio = h - (float)(hs);
      float w_ratio = w - (float)(ws);

      atomicAdd(bottom_diff[roi_bi][c][hs][ws], top_diff[n][c][ph][pw] * (1. - h_ratio) * (1 - w_ratio));
      atomicAdd(bottom_diff[roi_bi][c][hs][ws + 1], top_diff[n][c][ph][pw] * (1. - h_ratio) * w_ratio);
      atomicAdd(bottom_diff[roi_bi][c][hs + 1][ws], top_diff[n][c][ph][pw] * h_ratio * (1 - w_ratio));
      atomicAdd(bottom_diff[roi_bi][c][hs + 1][ws + 1], top_diff[n][c][ph][pw] * h_ratio * w_ratio);
    }
  }
}

int ROIAlignBackwardLaucher(const torch::Tensor top_diff,
                            const float spatial_scale,
                            const int aligned_height,
                            const int aligned_width,
                            const torch::Tensor rois,
                            torch::Tensor bottom_diff,
                            cudaStream_t stream) {
  const int kThreadsPerBlock = 1024;
  // Number of ROIs
  int num_rois = rois.size(0);
  // data height
  int height = bottom_diff.size(2);
  // data width
  int width = bottom_diff.size(3);
  // Number of channels
  int channels = bottom_diff.size(1);

  const int output_size = num_rois * aligned_height * aligned_width * channels;
  cudaError_t err;

  ROIAlignBackward <<< (output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream >>> (
    output_size, top_diff, spatial_scale, height, width, channels,
    aligned_height, aligned_width, bottom_diff, rois);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return 1;
}


#ifdef __cplusplus
}
#endif
