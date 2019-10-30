from torch.autograd import Function
from . import _roi_align as roi_align


# TODO use save_for_backward instead
class RoIAlignFunction(Function):

  aligned_width = 0
  aligned_height = 0
  spatial_scale = 0

  def __init__(self, aligned_height, aligned_width, spatial_scale):
    RoIAlignFunction.static_init(aligned_height, aligned_width, spatial_scale)

  @staticmethod
  def static_init(aligned_height, aligned_width, spatial_scale):
    RoIAlignFunction.aligned_width = int(aligned_width)
    RoIAlignFunction.aligned_height = int(aligned_height)
    RoIAlignFunction.spatial_scale = float(spatial_scale)
    pass

  @staticmethod
  def forward(ctx, features, rois):
    ctx.save_for_backward(features, rois)
    batch_size, num_channels, data_height, data_width = features.size()
    num_rois = rois.size(0)

    output = features.new(num_rois, num_channels,
                          RoIAlignFunction.aligned_height,
                          RoIAlignFunction.aligned_width).zero_()
    if features.is_cuda:
      roi_align.roi_align_forward_gpu(RoIAlignFunction.aligned_height,
                                      RoIAlignFunction.aligned_width,
                                      RoIAlignFunction.spatial_scale,
                                      features,
                                      rois,
                                      output)
    else:
      roi_align.roi_align_forward(ctx.aligned_height,
                                  ctx.aligned_width,
                                  ctx.spatial_scale,
                                  features,
                                  rois,
                                  output)

    return output

  @staticmethod
  def backward(ctx, grad_output):
    features, rois = ctx.saved_tensors
    feature_size = features.size()
    assert(feature_size is not None and grad_output.is_cuda)

    batch_size, num_channels, data_height, data_width = feature_size

    grad_input = rois.new(batch_size, num_channels, data_height,
                          data_width).zero_()
    roi_align.roi_align_backward_gpu(RoIAlignFunction.aligned_height,
                                     RoIAlignFunction.aligned_width,
                                     RoIAlignFunction.spatial_scale,
                                     grad_output,
                                     rois,
                                     grad_input)

    # print grad_input

    return grad_input, None
