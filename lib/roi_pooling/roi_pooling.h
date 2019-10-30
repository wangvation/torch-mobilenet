int roi_pooling_forward(int pooled_height,
                        int pooled_width,
                        float spatial_scale,
                        torch::Tensor features,
                        torch::Tensor rois,
                        torch::Tensor output);