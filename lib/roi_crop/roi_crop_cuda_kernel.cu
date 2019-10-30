#include <stdbool.h>
#include <stdio.h>
#include "roi_crop_cuda_kernel.h"

#define real float

// Bilinear sampling is done in BHWD (coalescing is not obvious in BDHW)
// we assume BHWD format in inputImages
// we assume BHW(YX) format on grids

__device__ void getTopLeft(float x, int width, int& point, float& weight)
{
  /* for interpolation :
     stores in point and weight :
     - the x-coordinate of the pixel on the left (or y-coordinate of the upper pixel)
     - the weight for interpolating
  */

  float xcoord = (x + 1) * (width - 1) / 2;
  point = floor(xcoord);
  weight = 1 - (xcoord - point);
}

__device__ bool between(int value, int lowerBound, int upperBound)
{
  return (value >= lowerBound && value <= upperBound);
}

__device__ void sumReduceShMem(volatile float s[])
{
  /* obviously only works for 32 elements */
  /* sums up a shared memory array of 32 elements, stores it in s[0] */
  /* whole warp can then read first element (broadcasting) */
  if (threadIdx.x < 16) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x + 16]; }
  if (threadIdx.x < 8) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x + 8]; }
  if (threadIdx.x < 4) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x + 4]; }
  if (threadIdx.x < 2) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x + 2]; }
  if (threadIdx.x < 1) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x + 1]; }
}

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

__global__ void bilinearSamplingFromGrid(const int nthreads,
                                         float* inputImages_data,
                                         int inputImages_strideBatch,
                                         int inputImages_strideChannels,
                                         int inputImages_strideHeight,
                                         int inputImages_strideWidth,
                                         float* grids_data,
                                         int grids_strideBatch,
                                         int grids_strideYX,
                                         int grids_strideHeight,
                                         int grids_strideWidth,
                                         float* output_data,
                                         int output_strideBatch,
                                         int output_strideChannels,
                                         int output_strideHeight,
                                         int output_strideWidth,
                                         int inputImages_channels,
                                         int inputImages_height,
                                         int inputImages_width,
                                         int output_batchsize,
                                         int output_channels,
                                         int output_height,
                                         int output_width,
                                         int roiPerImage)
{
  CUDA_KERNEL_LOOP(index, nthreads)
  {
    const int xOut = index % output_width;
    const int yOut = (index / output_width) % output_height;
    const int cOut  = (index / output_width / output_height) % output_channels;
    const int b = index / output_width / output_height / output_channels;

    const int width = inputImages_width;
    const int height = inputImages_height;

    const int b_input = b / roiPerImage;

    float yf = grids_data[b * grids_strideBatch + yOut * grids_strideHeight + xOut * grids_strideWidth];
    float xf = grids_data[b * grids_strideBatch + yOut * grids_strideHeight + xOut * grids_strideWidth + 1];

    int yInTopLeft, xInTopLeft;
    float yWeightTopLeft, xWeightTopLeft;
    getTopLeft(xf, inputImages_width, xInTopLeft, xWeightTopLeft);
    getTopLeft(yf, inputImages_height, yInTopLeft, yWeightTopLeft);

    // const int outAddress = output_strideBatch * b + output_strideHeight * yOut + output_strideWidth * xOut;
    const int outAddress = output_strideBatch * b + output_strideChannels * cOut + output_strideHeight * yOut + xOut;

    const int inTopLeftAddress = inputImages_strideBatch * b_input + inputImages_strideChannels * cOut + inputImages_strideHeight * yInTopLeft + xInTopLeft;
    const int inTopRightAddress = inTopLeftAddress + inputImages_strideWidth;
    const int inBottomLeftAddress = inTopLeftAddress + inputImages_strideHeight;
    const int inBottomRightAddress = inBottomLeftAddress + inputImages_strideWidth;

    float v = 0;
    float inTopLeft = 0;
    float inTopRight = 0;
    float inBottomLeft = 0;
    float inBottomRight = 0;

    bool topLeftIsIn = between(xInTopLeft, 0, width - 1) && between(yInTopLeft, 0, height - 1);
    bool topRightIsIn = between(xInTopLeft + 1, 0, width - 1) && between(yInTopLeft, 0, height - 1);
    bool bottomLeftIsIn = between(xInTopLeft, 0, width - 1) && between(yInTopLeft + 1, 0, height - 1);
    bool bottomRightIsIn = between(xInTopLeft + 1, 0, width - 1) && between(yInTopLeft + 1, 0, height - 1);

    if (!topLeftIsIn && !topRightIsIn && !bottomLeftIsIn && !bottomRightIsIn)
      continue;

    if (topLeftIsIn) inTopLeft = inputImages_data[inTopLeftAddress];
    if (topRightIsIn) inTopRight = inputImages_data[inTopRightAddress];
    if (bottomLeftIsIn) inBottomLeft = inputImages_data[inBottomLeftAddress];
    if (bottomRightIsIn) inBottomRight = inputImages_data[inBottomRightAddress];

    v = xWeightTopLeft * yWeightTopLeft * inTopLeft
        + (1 - xWeightTopLeft) * yWeightTopLeft * inTopRight
        + xWeightTopLeft * (1 - yWeightTopLeft) * inBottomLeft
        + (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * inBottomRight;

    output_data[outAddress] = v;
  }

}

__global__ void backwardBilinearSampling(const int nthreads,
                                         float* inputImages_data,
                                         int inputImages_strideBatch,
                                         int inputImages_strideChannels,
                                         int inputImages_strideHeight,
                                         int inputImages_strideWidth,
                                         float* gradInputImages_data,
                                         int gradInputImages_strideBatch,
                                         int gradInputImages_strideChannels,
                                         int gradInputImages_strideHeight,
                                         int gradInputImages_strideWidth,
                                         float* grids_data,
                                         int grids_strideBatch,
                                         int grids_strideYX,
                                         int grids_strideHeight,
                                         int grids_strideWidth,
                                         float* gradGrids_data,
                                         int gradGrids_strideBatch,
                                         int gradGrids_strideYX,
                                         int gradGrids_strideHeight,
                                         int gradGrids_strideWidth,
                                         float* gradOutput_data,
                                         int gradOutput_strideBatch,
                                         int gradOutput_strideChannels,
                                         int gradOutput_strideHeight,
                                         int gradOutput_strideWidth,
                                         int inputImages_channels,
                                         int inputImages_height,
                                         int inputImages_width,
                                         int gradOutput_batchsize,
                                         int gradOutput_channels,
                                         int gradOutput_height,
                                         int gradOutput_width,
                                         int roiPerImage)
{

  CUDA_KERNEL_LOOP(index, nthreads)
  {
    const int xOut = index % gradOutput_width;
    const int yOut = (index / gradOutput_width) % gradOutput_height;
    const int cOut  = (index / gradOutput_width / gradOutput_height) % gradOutput_channels;
    const int b = index / gradOutput_width / gradOutput_height / gradOutput_channels;

    const int b_input = b / roiPerImage;

    const int width = inputImages_width;
    const int height = inputImages_height;

    float yf = grids_data[b * grids_strideBatch + yOut * grids_strideHeight + xOut * grids_strideWidth];
    float xf = grids_data[b * grids_strideBatch + yOut * grids_strideHeight + xOut * grids_strideWidth + 1];

    int yInTopLeft, xInTopLeft;
    float yWeightTopLeft, xWeightTopLeft;
    getTopLeft(xf, inputImages_width, xInTopLeft, xWeightTopLeft);
    getTopLeft(yf, inputImages_height, yInTopLeft, yWeightTopLeft);

    const int inTopLeftAddress = inputImages_strideBatch * b_input + inputImages_strideChannels * cOut + inputImages_strideHeight * yInTopLeft + xInTopLeft;
    const int inTopRightAddress = inTopLeftAddress + inputImages_strideWidth;
    const int inBottomLeftAddress = inTopLeftAddress + inputImages_strideHeight;
    const int inBottomRightAddress = inBottomLeftAddress + inputImages_strideWidth;

    const int gradInputImagesTopLeftAddress = gradInputImages_strideBatch * b_input + gradInputImages_strideChannels * cOut
                                              + gradInputImages_strideHeight * yInTopLeft + xInTopLeft;
    const int gradInputImagesTopRightAddress = gradInputImagesTopLeftAddress + gradInputImages_strideWidth;
    const int gradInputImagesBottomLeftAddress = gradInputImagesTopLeftAddress + gradInputImages_strideHeight;
    const int gradInputImagesBottomRightAddress = gradInputImagesBottomLeftAddress + gradInputImages_strideWidth;

    const int gradOutputAddress = gradOutput_strideBatch * b + gradOutput_strideChannels * cOut + gradOutput_strideHeight * yOut + xOut;

    float topLeftDotProduct = 0;
    float topRightDotProduct = 0;
    float bottomLeftDotProduct = 0;
    float bottomRightDotProduct = 0;

    bool topLeftIsIn = between(xInTopLeft, 0, width - 1) && between(yInTopLeft, 0, height - 1);
    bool topRightIsIn = between(xInTopLeft + 1, 0, width - 1) && between(yInTopLeft, 0, height - 1);
    bool bottomLeftIsIn = between(xInTopLeft, 0, width - 1) && between(yInTopLeft + 1, 0, height - 1);
    bool bottomRightIsIn = between(xInTopLeft + 1, 0, width - 1) && between(yInTopLeft + 1, 0, height - 1);

    float gradOutValue = gradOutput_data[gradOutputAddress];
    // bool between(int value, int lowerBound, int upperBound)
    if (topLeftIsIn)
    {
      float inTopLeft = inputImages_data[inTopLeftAddress];
      topLeftDotProduct += inTopLeft * gradOutValue;
      atomicAdd(&gradInputImages_data[gradInputImagesTopLeftAddress], xWeightTopLeft * yWeightTopLeft * gradOutValue);
    }

    if (topRightIsIn)
    {
      float inTopRight = inputImages_data[inTopRightAddress];
      topRightDotProduct += inTopRight * gradOutValue;
      atomicAdd(&gradInputImages_data[gradInputImagesTopRightAddress], (1 - xWeightTopLeft) * yWeightTopLeft * gradOutValue);
    }

    if (bottomLeftIsIn)
    {
      float inBottomLeft = inputImages_data[inBottomLeftAddress];
      bottomLeftDotProduct += inBottomLeft * gradOutValue;
      atomicAdd(&gradInputImages_data[gradInputImagesBottomLeftAddress], xWeightTopLeft * (1 - yWeightTopLeft) * gradOutValue);
    }

    if (bottomRightIsIn)
    {
      float inBottomRight = inputImages_data[inBottomRightAddress];
      bottomRightDotProduct += inBottomRight * gradOutValue;
      atomicAdd(&gradInputImages_data[gradInputImagesBottomRightAddress], (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * gradOutValue);
    }
  }
}


int BilinearSamplerBHWD_updateOutput_cuda_kernel(int output_batchsize,
                                                 int output_channels,
                                                 int output_height,
                                                 int output_width,
                                                 int inputImages_batchsize,
                                                 int inputImages_channels,
                                                 int inputImages_height,
                                                 int inputImages_width,
                                                 float* inputImages_data,
                                                 int inputImages_strideBatch,
                                                 int inputImages_strideChannels,
                                                 int inputImages_strideHeight,
                                                 int inputImages_strideWidth,
                                                 float* grids_data,
                                                 int grids_strideBatch,
                                                 int grids_strideYX,
                                                 int grids_strideHeight,
                                                 int grids_strideWidth,
                                                 float* output_data,
                                                 int output_strideBatch,
                                                 int output_strideChannels,
                                                 int output_strideHeight,
                                                 int output_strideWidth)
{
  const int kThreadsPerBlock = 1024;
  int output_size = output_batchsize * output_height * output_width * output_channels;
  int roiPerImage = output_batchsize / inputImages_batchsize;

  // printf("forward pass\n");
  int blocks = (output_size + kThreadsPerBlock - 1) / kThreadsPerBlock;
  bilinearSamplingFromGrid <<< blocks, kThreadsPerBlock>>>(output_size,
                                                           inputImages_data,
                                                           inputImages_strideBatch,
                                                           inputImages_strideChannels,
                                                           inputImages_strideHeight,
                                                           inputImages_strideWidth,
                                                           grids_data,
                                                           grids_strideBatch,
                                                           grids_strideYX,
                                                           grids_strideHeight,
                                                           grids_strideWidth,
                                                           output_data,
                                                           output_strideBatch,
                                                           output_strideChannels,
                                                           output_strideHeight,
                                                           output_strideWidth,
                                                           inputImages_channels,
                                                           inputImages_height,
                                                           inputImages_width,
                                                           output_channels,
                                                           output_height,
                                                           output_width,
                                                           output_batchsize,
                                                           roiPerImage);

  return 0;
}

int BilinearSamplerBHWD_updateGradInput_cuda_kernel(int gradOutput_batchsize,
                                                    int gradOutput_channels,
                                                    int gradOutput_height,
                                                    int gradOutput_width,
                                                    int inputImages_batch ,
                                                    int inputImages_channels ,
                                                    int inputImages_height ,
                                                    int inputImages_width ,
                                                    float* inputImages_data,
                                                    int inputImages_strideBatch,
                                                    int inputImages_strideChannels,
                                                    int inputImages_strideHeight,
                                                    int inputImages_strideWidth,
                                                    float* grids_data,
                                                    int grids_strideBatch,
                                                    int grids_strideYX,
                                                    int grids_strideHeight,
                                                    int grids_strideWidth,
                                                    float* gradInputImages_data,
                                                    int gradInputImages_strideBatch,
                                                    int gradInputImages_strideChannels,
                                                    int gradInputImages_strideHeight,
                                                    int gradInputImages_strideWidth,
                                                    float* gradGrids_data,
                                                    int gradGrids_strideBatch,
                                                    int gradGrids_strideYX,
                                                    int gradGrids_strideHeight,
                                                    int gradGrids_strideWidth,
                                                    float* gradOutput_data,
                                                    int gradOutput_strideBatch,
                                                    int gradOutput_strideChannels,
                                                    int gradOutput_strideHeight,
                                                    int gradOutput_strideWidth)
{

  const int kThreadsPerBlock = 1024;
  int output_size = gradOutput_batchsize * gradOutput_channels * gradOutput_height * gradOutput_width;
  int roiPerImage = gradOutput_batchsize / inputImages_batch;


  int blocks = (output_size + kThreadsPerBlock - 1) / kThreadsPerBlock;
  backwardBilinearSampling <<< blocks, kThreadsPerBlock>>>(output_size,
                                                           inputImages_data,
                                                           inputImages_strideBatch,
                                                           inputImages_strideChannels,
                                                           inputImages_strideHeight,
                                                           inputImages_strideWidth,
                                                           gradInputImages_data,
                                                           gradInputImages_strideBatch,
                                                           gradInputImages_strideChannels,
                                                           gradInputImages_strideHeight,
                                                           gradInputImages_strideWidth,
                                                           grids_data,
                                                           grids_strideBatch,
                                                           grids_strideYX,
                                                           grids_strideHeight,
                                                           grids_strideWidth,
                                                           gradGrids_data,
                                                           gradGrids_strideBatch,
                                                           gradGrids_strideYX,
                                                           gradGrids_strideHeight,
                                                           gradGrids_strideWidth,
                                                           gradOutput_data,
                                                           gradOutput_strideBatch,
                                                           gradOutput_strideChannels,
                                                           gradOutput_strideHeight,
                                                           gradOutput_strideWidth,
                                                           inputImages_channels,
                                                           inputImages_height,
                                                           inputImages_width,
                                                           gradOutput_channels,
                                                           gradOutput_height,
                                                           gradOutput_width,
                                                           gradOutput_batchsize,
                                                           roiPerImage);
  return 0;
}

