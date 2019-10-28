#ifndef _ROI_ALIGN_KERNEL
#define _ROI_ALIGN_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

__global__ void ROIAlignForward(const int nthreads,
                                const torch::Tensor bottom_data,
                                const float spatial_scale,
                                const int height,
                                const int width,
                                const int channels,
                                const int aligned_height,
                                const int aligned_width,
                                const torch::Tensor bottom_rois,
                                torch::Tensor top_data);

int ROIAlignForwardLaucher(const torch::Tensor bottom_data,
                           const float spatial_scale,
                           const int aligned_height,
                           const int aligned_width,
                           const torch::Tensor bottom_rois,
                           torch::Tensor top_data,
                           cudaStream_t stream);

__global__ void ROIAlignBackward(const int nthreads,
                                 const torch::Tensor top_diff,
                                 const float spatial_scale,
                                 const int height,
                                 const int width,
                                 const int channels,
                                 const int aligned_height,
                                 const int aligned_width,
                                 torch::Tensor bottom_diff,
                                 const torch::Tensor bottom_rois);

int ROIAlignBackwardLaucher(const torch::Tensor top_diff,
                            const float spatial_scale,
                            const int aligned_height,
                            const int aligned_width,
                            const torch::Tensor bottom_rois,
                            torch::Tensor bottom_diff,
                            cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

