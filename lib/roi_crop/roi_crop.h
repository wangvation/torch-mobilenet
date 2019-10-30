int BilinearSamplerBHWD_updateOutput(torch::Tensor inputImages,
                                     torch::Tensor grids,
                                     torch::Tensor output);

int BilinearSamplerBHWD_updateGradInput(torch::Tensor inputImages,
                                        torch::Tensor grids,
                                        torch::Tensor gradInputImages,
                                        torch::Tensor gradGrids,
                                        torch::Tensor gradOutput);



int BilinearSamplerBCHW_updateOutput(torch::Tensor inputImages,
                                     torch::Tensor grids,
                                     torch::Tensor output);

int BilinearSamplerBCHW_updateGradInput(torch::Tensor inputImages,
                                        torch::Tensor grids,
                                        torch::Tensor gradInputImages,

                                        torch::Tensor gradGrids,
                                        torch::Tensor gradOutput);
