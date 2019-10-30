import torch
from torch.autograd import Function
from . import _roi_pooling as roi_pooling
import pdb


class RoIPoolFunction(Function):
  pooled_width = 0
  pooled_height = 0
  spatial_scale = 0

  def __init__(ctx, pooled_height, pooled_width, spatial_scale):
    RoIPoolFunction.static_init(pooled_height, pooled_width, spatial_scale)

  def static_init(pooled_height, pooled_width, spatial_scale):
    RoIPoolFunction.pooled_width = pooled_width
    RoIPoolFunction.pooled_height = pooled_height
    RoIPoolFunction.spatial_scale = spatial_scale

  @staticmethod
  def forward(ctx, features, rois):
    feature_size = features.size()
    batch_size, num_channels, data_height, data_width = feature_size
    num_rois = rois.size(0)
    output = features.new(num_rois,
                          num_channels,
                          RoIPoolFunction.pooled_height,
                          RoIPoolFunction.pooled_width).zero_()
    argmax = features.new(num_rois, num_channels,
                          RoIPoolFunction.pooled_height,
                          RoIPoolFunction.pooled_width).zero_().int()
    ctx.save_for_backward(features, rois, argmax)
    rois = rois
    if not features.is_cuda:
      _features = features.permute(0, 2, 3, 1)
      roi_pooling.roi_pooling_forward(RoIPoolFunction.pooled_height,
                                      RoIPoolFunction.pooled_width,
                                      RoIPoolFunction.spatial_scale,
                                      _features,
                                      rois,
                                      output)
    else:
      roi_pooling.roi_pooling_forward_cuda(RoIPoolFunction.pooled_height,
                                           RoIPoolFunction.pooled_width,
                                           RoIPoolFunction.spatial_scale,
                                           features,
                                           rois,
                                           output,
                                           argmax)

    return output

  @staticmethod
  def backward(ctx, grad_output):
    features, rois, argmax = ctx.saved_tensors
    feature_size = features.size()
    assert(feature_size is not None and grad_output.is_cuda)
    batch_size, num_channels, data_height, data_width = feature_size
    grad_input = grad_output.new(
        batch_size, num_channels, data_height, data_width).zero_()

    roi_pooling.roi_pooling_backward_cuda(RoIPoolFunction.pooled_height,
                                          RoIPoolFunction.pooled_width,
                                          RoIPoolFunction.spatial_scale,
                                          grad_output,
                                          rois,
                                          grad_input,
                                          argmax)

    return grad_input, None
