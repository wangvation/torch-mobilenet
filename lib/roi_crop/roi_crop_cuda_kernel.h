#ifdef __cplusplus
extern "C" {
#endif


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
                                                 int output_strideWidth);

int BilinearSamplerBHWD_updateGradInput_cuda_kernel(int gradOutput_batch,
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
                                                    int gradOutput_strideWidth);

#ifdef __cplusplus
}
#endif
