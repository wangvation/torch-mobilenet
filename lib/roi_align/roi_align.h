int roi_align_forward(int aligned_height,
                      int aligned_width,
                      float spatial_scale,
                      torch::Tensor features,
                      torch::Tensor rois,
                      torch::Tensor output);

int roi_align_backward(int aligned_height,
                       int aligned_width,
                       float spatial_scale,
                       torch::Tensor top_grad,
                       torch::Tensor rois,
                       torch::Tensor bottom_grad);

int roi_align_forward_gpu(int aligned_height,
                          int aligned_width,
                          float spatial_scale,
                          torch::Tensor features,
                          torch::Tensor rois,
                          torch::Tensor output);

int roi_align_backward_gpu(int aligned_height,
                           int aligned_width,
                           float spatial_scale,
                           torch::Tensor top_grad,
                           torch::Tensor rois,
                           torch::Tensor bottom_grad);
