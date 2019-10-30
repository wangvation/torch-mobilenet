// Bilinear sampling is done in BHWD (coalescing is not obvious in BDHW)
// we assume BHWD format in inputImages
// we assume BHW(YX) format on grids

int BilinearSamplerBHWD_updateOutput_cuda(torch::Tensor inputImages,
                                          torch::Tensor grids,
                                          torch::Tensor output);

int BilinearSamplerBHWD_updateGradInput_cuda(torch::Tensor inputImages,
                                             torch::Tensor grids,
                                             torch::Tensor gradInputImages,
                                             torch::Tensor gradGrids,
                                             torch::Tensor gradOutput);
