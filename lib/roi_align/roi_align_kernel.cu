#include <stdio.h>
#include <math.h>
#include <float.h>
#include <iostream>
#include "roi_align_kernel.h"

using namespace std;

template <typename scalar_t>
__global__ void ROIAlignForward(const int nthreads,
                                const float spatial_scale,
                                const int height,
                                const int width,
                                const int channels,
                                const int aligned_height,
                                const int aligned_width,
                                const scalar_t* __restrict__  bottom_data,
                                const scalar_t* __restrict__  bottom_rois,
                                scalar_t* __restrict__  top_data) {

  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < nthreads;
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
    int roi_bi = floor(bottom_rois[n * 5 + 0]);
    scalar_t roi_start_w = bottom_rois[n * 5 + 1] * spatial_scale;
    scalar_t roi_start_h = bottom_rois[n * 5 + 2] * spatial_scale;
    scalar_t roi_end_w = bottom_rois[n * 5 + 3] * spatial_scale;
    scalar_t roi_end_h = bottom_rois[n * 5 + 4] * spatial_scale;

    // Force malformed ROIs to be 1x1
    scalar_t roi_width = fmaxf(roi_end_w - roi_start_w + 1., 0.);
    scalar_t roi_height = fmaxf(roi_end_h - roi_start_h + 1., 0.);
    scalar_t bin_size_h = roi_height / (aligned_height - 1.);
    scalar_t bin_size_w = roi_width / (aligned_width - 1.);

    scalar_t h = (scalar_t)(ph) * bin_size_h + roi_start_h;
    scalar_t w = (scalar_t)(pw) * bin_size_w + roi_start_w;

    int hs = fminf(floor(h), height - 2);
    int ws = fminf(floor(w), width - 2);

    int img_start = roi_bi * channels * height * width;

    // bilinear interpolation
    if (h < 0 || h >= height || w < 0 || w >= width) {
      top_data[index] = 0.;
    } else {
      scalar_t h_ratio = h - (scalar_t)(hs);
      scalar_t w_ratio = w - (scalar_t)(ws);
      int upleft = img_start + (c * height + hs) * width + ws;
      int upright = upleft + 1;
      int downleft = upleft + width;
      int downright = downleft + 1;

      top_data[index] = 0.;
      top_data[index] += bottom_data[upleft] * (1. - h_ratio) * (1. - w_ratio);
      top_data[index] += bottom_data[upright] * (1. - h_ratio) * w_ratio;
      top_data[index] += bottom_data[downleft] * h_ratio * (1. - w_ratio);
      top_data[index] += bottom_data[downright] * h_ratio * w_ratio;
    }
  }
}

void ROIAlignForwardLaucher(const float spatial_scale,
                           const int aligned_height,
                           const int aligned_width,
                           const int num_rois,
                           const int height,
                           const int width,
                           const int channels,
                           const float* __restrict__ features,
                           const float* __restrict__ rois,
                           float* __restrict__ output) {
  const int kThreadsPerBlock = 1024;

  const int output_size = num_rois * aligned_height * aligned_width * channels;

  const int block_size = (int)ceil(output_size * 1.0 / kThreadsPerBlock) ;
  cout << "forward blocks:" << block_size << endl;

  ROIAlignForward<float> <<< block_size, kThreadsPerBlock >>>(
    output_size,
    spatial_scale,
    height,
    width,
    channels,
    aligned_height,
    aligned_width,
    features,
    rois,
    output);

  // cudaError_t err;
  // err = cudaGetLastError();
  // if (cudaSuccess != err) {
  //   fprintf( stderr, "0cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
  //   // exit( -1 );
  // }

}

template <typename scalar_t>
__global__ void ROIAlignBackward(const int nthreads,
                                 const float spatial_scale,
                                 const int height,
                                 const int width,
                                 const int channels,
                                 const int aligned_height,
                                 const int aligned_width,
                                 const scalar_t* __restrict__ top_diff,
                                 const scalar_t* __restrict__ bottom_rois,
                                 scalar_t* __restrict__ bottom_diff) {

  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < nthreads;
       index += blockDim.x * gridDim.x) {

    // (n, c, ph, pw) is an element in the aligned output
    int pw = index % aligned_width;
    int ph = (index / aligned_width) % aligned_height;
    int c  = (index / aligned_width / aligned_height) % channels;
    int n  = index / aligned_width / aligned_height / channels;

    int roi_bi = (int)bottom_rois[n * 5 + 0];
    scalar_t roi_start_w = bottom_rois[n * 5 + 1] * spatial_scale;
    scalar_t roi_start_h = bottom_rois[n * 5 + 2] * spatial_scale;
    scalar_t roi_end_w = bottom_rois[n * 5 + 3] * spatial_scale;
    scalar_t roi_end_h = bottom_rois[n * 5 + 4] * spatial_scale;

    // Force malformed ROIs to be 1x1
    scalar_t roi_width = fmaxf(roi_end_w - roi_start_w + 1., 0.);
    scalar_t roi_height = fmaxf(roi_end_h - roi_start_h + 1., 0.);
    scalar_t bin_size_h = roi_height / (aligned_height - 1.);
    scalar_t bin_size_w = roi_width / (aligned_width - 1.);

    scalar_t h = (scalar_t)(ph) * bin_size_h + roi_start_h;
    scalar_t w = (scalar_t)(pw) * bin_size_w + roi_start_w;

    int hs = fminf(floor(h), height - 2);
    int ws = fminf(floor(w), width - 2);

    int img_start = roi_bi * channels * height * width;
    // bilinear interpolation
    if (h >= 0 && h < height && w >= 0 && w < width) {
      scalar_t h_ratio = h - (scalar_t)(hs);
      scalar_t w_ratio = w - (scalar_t)(ws);

      int upleft = img_start + (c * height + hs) * width + ws;
      int upright = upleft + 1;
      int downleft = upleft + width;
      int downright = downleft + 1;

      bottom_diff[upleft] += top_diff[index] * (1. - h_ratio) * (1 - w_ratio);
      bottom_diff[upright] += top_diff[index] * (1. - h_ratio) * w_ratio;
      bottom_diff[downleft] += top_diff[index] * h_ratio * (1 - w_ratio);
      bottom_diff[downright] += top_diff[index] * h_ratio * w_ratio;
    }
  }
}


void ROIAlignBackwardLaucher(const float spatial_scale,
                            const int aligned_height,
                            const int aligned_width,
                            const int num_rois,
                            const int height,
                            const int width,
                            const int channels,
                            const float* __restrict__ top_diff,
                            const float* __restrict__ rois,
                            float* __restrict__ bottom_diff) {
  const int kThreadsPerBlock = 1024;

  const int output_size = num_rois * aligned_height * aligned_width * channels;
  const int block_size = (int)ceil(output_size * 1.0 / kThreadsPerBlock) ;
  cout << "backward blocks: " << block_size << endl;

  ROIAlignBackward<float> <<< block_size, kThreadsPerBlock >>> (
    output_size,  spatial_scale, height, width,
    channels, aligned_height, aligned_width,
    top_diff, rois, bottom_diff);

  // cudaError_t err;
  // err = cudaGetLastError();
  // if (cudaSuccess != err) {
  //   fprintf( stderr, "1cudaCheckError() failed : %s\n", cudaGetErrorString(err));
  //   // exit( -1 );
  // }

}