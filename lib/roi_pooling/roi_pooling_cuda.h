int roi_pooling_forward_cuda(int pooled_height,
                             int pooled_width,
                             float spatial_scale,
                             torch::Tensor features,
                             torch::Tensor rois,
                             torch::Tensor output,
                             torch::Tensor argmax);

int roi_pooling_backward_cuda(int pooled_height,
                              int pooled_width,
                              float spatial_scale,
                              torch::Tensor top_grad,
                              torch::Tensor rois,
                              torch::Tensor bottom_grad,
                              torch::Tensor argmax);