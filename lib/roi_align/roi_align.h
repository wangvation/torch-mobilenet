int roi_align_forward(int aligned_height,
                      int aligned_width,
                      float spatial_scale,
                      THFloatTensor * features,
                      THFloatTensor * rois,
                      THFloatTensor * output);

int roi_align_backward(int aligned_height,
                       int aligned_width,
                       float spatial_scale,
                       THFloatTensor * top_grad,
                       THFloatTensor * rois,
                       THFloatTensor * bottom_grad);

int roi_align_forward_gpu(int aligned_height,
                          int aligned_width,
                          float spatial_scale,
                          THCudaTensor * features,
                          THCudaTensor * rois,
                          THCudaTensor * output);

int roi_align_backward_gpu(int aligned_height,
                           int aligned_width,
                           float spatial_scale,
                           THCudaTensor * top_grad,
                           THCudaTensor * rois,
                           THCudaTensor * bottom_grad);

// torch::Tensor roi_align_cuda_test(torch::Tensor a,
//                                   torch::Tensor b);