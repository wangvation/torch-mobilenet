// int nms_cuda(THCudaTensor *keep_out, THCudaTensor *num_out,
//             THCudaTensor *boxes_host, THCudaTensor *nms_overlap_thresh);

int nms_cuda(torch::Tensor keep_out,
             torch::Tensor boxes_host,
             torch::Tensor num_out,
             float nms_overlap_thresh);
