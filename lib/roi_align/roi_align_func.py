import torch
from torch.autograd import Function
from . import _roi_align as roi_align


# TODO use save_for_backward instead
class RoIAlignFunction(Function):
  def __init__(self, aligned_height, aligned_width, spatial_scale):
    self.aligned_width = int(aligned_width)
    self.aligned_height = int(aligned_height)
    self.spatial_scale = float(spatial_scale)
    self.rois = None
    self.feature_size = None

  def forward(self, features, rois):
    self.rois = rois
    self.feature_size = features.size()
    a = torch.full((3, 3), 2, dtype=torch.float64).cuda()
    b = torch.full((3, 3), 3, dtype=torch.float64).cuda()
    c = roi_align.roi_align_cuda_test(a, b)
    print(features.dtype)
    print(rois.dtype)
    print(c.size(), c, c.dtype)

    batch_size, num_channels, data_height, data_width = features.size()
    num_rois = rois.size(0)

    print('aligned features.')
    output = features.new(num_rois, num_channels,
                          self.aligned_height,
                          self.aligned_width).zero_()
    print('aligned is_cuda.')
    if features.is_cuda:
      roi_align.roi_align_forward_gpu(self.aligned_height,
                                      self.aligned_width,
                                      self.spatial_scale,
                                      features,
                                      rois,
                                      output)
    else:
      roi_align.roi_align_forward(self.aligned_height,
                                  self.aligned_width,
                                  self.spatial_scale,
                                  features,
                                  rois,
                                  output)

    print('aligned finished.')

    return output

  def backward(self, grad_output):
    assert(self.feature_size is not None and grad_output.is_cuda)

    batch_size, num_channels, data_height, data_width = self.feature_size

    grad_input = self.rois.new(batch_size, num_channels, data_height,
                               data_width).zero_()
    roi_align.roi_align_backward_gpu(self.aligned_height,
                                     self.aligned_width,
                                     self.spatial_scale,
                                     grad_output,
                                     self.rois,
                                     grad_input)

    # print grad_input

    return grad_input, None
