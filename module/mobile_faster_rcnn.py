# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .faster_rcnn import _fasterRCNN


class MobileFasterRCNN(_fasterRCNN):
  def __init__(self,
               mobile_net,
               num_classes,
               classes,
               dout_base_model=512,
               model_path=None,
               class_agnostic=False):
    self.model_path = model_path
    self.dout_base_model = dout_base_model
    self.class_agnostic = class_agnostic
    _fasterRCNN.__init__(self, classes, class_agnostic)
    self.mobile_net = mobile_net

  def _init_modules(self):

    if self.model_path:
      print("Loading pretrained weights from %s" % (self.model_path))
      state_dict = torch.load(self.model_path)
      self.mobile_net.load_state_dict({k: v for k, v in state_dict.items()
                                       if k in self.mobile_net.state_dict()
                                       })

    self.mobile_net.classifier = nn.Sequential(
        *list(self.mobile_net.classifier._modules.values()))

    self.RCNN_base = nn.Sequential(
        *list(self.mobile_net.features._modules.values()))

    # Fix the layers before conv3:
    for layer in range(10):
      for p in self.RCNN_base[layer].parameters():
        p.requires_grad = False

    self.RCNN_top = self.mobile_net.classifier

    # not using the last maxpool layer
    self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(4096, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)

  def _head_to_tail(self, pool5):

    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7
