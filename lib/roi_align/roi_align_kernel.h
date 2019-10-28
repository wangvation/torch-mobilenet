#ifndef _ROI_ALIGN_KERNEL
#define _ROI_ALIGN_KERNEL


void ROIAlignForwardLaucher(const float spatial_scale,
                           const int aligned_height,
                           const int aligned_width,
                           const int num_rois,
                           const int height,
                           const int width,
                           const int channels,
                           const float* __restrict__ features,
                           const float* __restrict__ rois,
                           float* __restrict__ output);

void ROIAlignBackwardLaucher(const float spatial_scale,
                            const int aligned_height,
                            const int aligned_width,
                            const int num_rois,
                            const int height,
                            const int width,
                            const int channels,
                            const float* __restrict__ top_diff,
                            const float* __restrict__ rois,
                            float* __restrict__ bottom_diff);


#endif

