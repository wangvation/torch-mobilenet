from torch.nn.modules.module import Module
from .roi_pool_func import RoIPoolFunction


class _RoIPooling(Module):
  def __init__(self, pooled_height, pooled_width, spatial_scale):
    super(_RoIPooling, self).__init__()

    RoIPoolFunction.static_init(int(pooled_width),
                                int(pooled_height),
                                float(spatial_scale))

  def forward(self, features, rois):
    return RoIPoolFunction.apply(features, rois)
