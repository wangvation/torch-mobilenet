from torch.nn.modules.module import Module
from .roi_crop_func import RoICropFunction


class _RoICrop(Module):
  def __init__(self, layout='BHWD'):
    super(_RoICrop, self).__init__()

  def forward(self, input1, input2):
    return RoICropFunction()(input1, input2)
