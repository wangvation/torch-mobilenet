#include <torch/torch.h>
#include <torch/extension.h>
#include <stdbool.h>
#include <stdio.h>
#include "roi_crop_cuda_kernel.h"

#define real float


// Bilinear sampling is done in BHWD (coalescing is not obvious in BDHW)
// we assume BHWD format in inputImages
// we assume BHW(YX) format on grids

int BilinearSamplerBHWD_updateOutput_cuda(torch::Tensor inputImages,
                                          torch::Tensor grids,
                                          torch::Tensor output) {
//  THCState *state = getCutorchState(L);
//  torch::Tensor inputImages = (torch::Tensor )luaT_checkudata(L, 2, "torch.CudaTensor");
//  torch::Tensor grids = (torch::Tensor )luaT_checkudata(L, 3, "torch.CudaTensor");
//  torch::Tensor output = (torch::Tensor )luaT_checkudata(L, 4, "torch.CudaTensor");

  int success = 0;
  success = BilinearSamplerBHWD_updateOutput_cuda_kernel(
              /*output_batchsize*/output.size(0),
              /*output_channels*/output.size(3),
              /*output_height*/output.size(1),
              /*output_width*/output.size(2),
              /*inputImages_batchsize*/inputImages.size(0),
              /*inputImages_channels*/inputImages.size(3),
              /*inputImages_height*/inputImages.size(1),
              /*inputImages_width*/inputImages.size(2),
              /*inputImages_data*/inputImages.data_ptr<float>(),
              /*inputImages_strideBatch*/inputImages.stride(0),
              /*inputImages_strideChannels*/inputImages.stride(3),
              /*inputImages_strideHeight*/inputImages.stride(1),
              /*inputImages_strideWidth*/inputImages.stride(2),
              /*grids_data*/grids.data_ptr<float>(),
              /*grids_strideBatch*/grids.stride(0),
              /*grids_strideYX*/grids.stride(3),
              /*grids_strideHeight*/grids.stride(1),
              /*grids_strideWidth*/grids.stride(2),
              /*output_data*/output.data_ptr<float>(),
              /*output_strideBatch*/output.stride(0),
              /*output_strideChannels*/output.stride(3),
              /*output_strideHeight*/output.stride(1),
              /*output_strideWidth*/output.stride(2));


  return success;
}

int BilinearSamplerBHWD_updateGradInput_cuda(torch::Tensor inputImages,
                                             torch::Tensor grids,
                                             torch::Tensor gradInputImages,
                                             torch::Tensor gradGrids,
                                             torch::Tensor gradOutput)
{
//  THCState *state = getCutorchState(L);
//  torch::Tensor inputImages = (torch::Tensor )luaT_checkudata(L, 2, "torch.CudaTensor");
//  torch::Tensor grids = (torch::Tensor )luaT_checkudata(L, 3, "torch.CudaTensor");
//  torch::Tensor gradInputImages = (torch::Tensor )luaT_checkudata(L, 4, "torch.CudaTensor");
//  torch::Tensor gradGrids = (torch::Tensor )luaT_checkudata(L, 5, "torch.CudaTensor");
//  torch::Tensor gradOutput = (torch::Tensor )luaT_checkudata(L, 6, "torch.CudaTensor");

  int success = 0;
  success = BilinearSamplerBHWD_updateGradInput_cuda_kernel(
              /*gradOutput_batch*/gradOutput.size(0),
              /*gradOutput_channels*/gradOutput.size(3),
              /*gradOutput_height*/gradOutput.size(1),
              /*gradOutput_width*/gradOutput.size(2),
              /*inputImages_batch*/inputImages.size(0),
              /*inputImages_channels*/inputImages.size(3),
              /*inputImages_height*/inputImages.size(1),
              /*inputImages_width*/inputImages.size(2),
              /*inputImages_data*/inputImages.data_ptr<float>(),
              /*inputImages_strideBatch*/inputImages.stride(0),
              /*inputImages_strideChannels*/inputImages.stride(3),
              /*inputImages_strideHeight*/inputImages.stride(1),
              /*inputImages_strideWidth*/inputImages.stride(2),
              /*grids*/grids.data_ptr<float>(),
              /*grids_strideBatch*/grids.stride(0),
              /*grids_strideYX*/grids.stride(3),
              /*grids_strideHeight*/grids.stride(1),
              /*grids_strideWidth*/grids.stride(2),
              /*gradInputImages_data*/gradInputImages.data_ptr<float>(),
              /*gradInputImages_strideBatch*/gradInputImages.stride(0),
              /*gradInputImages_strideChannels*/gradInputImages.stride(3),
              /*gradInputImages_strideHeight*/gradInputImages.stride(1),
              /*gradInputImages_strideWidth*/gradInputImages.stride(2),
              /*gradGrids_data*/gradGrids.data_ptr<float>(),
              /*gradGrids_strideBatch*/gradGrids.stride(0),
              /*gradGrids_strideYX*/gradGrids.stride(3),
              /*gradGrids_strideHeight*/gradGrids.stride(1),
              /*gradGrids_strideWidth*/gradGrids.stride(2),
              /*gradOutput_data*/gradOutput.data_ptr<float>(),
              /*gradOutput_strideBatch*/gradOutput.stride(0),
              /*gradOutput_strideChannels*/gradOutput.stride(3),
              /*gradOutput_strideHeight*/gradOutput.stride(1),
              /*gradOutput_strideWidth*/gradOutput.stride(2));


  return success;
}
