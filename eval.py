
import sys
import json
import torch
import os
import numpy as np
from pycocotools.cocoeval import COCOeval

import matplotlib.pyplot as plt
from pycocotools.coco import COCO


def dump_to_json():
  pass


def eval(anno_json,
         result_json,
         anno_type):
  annType = ['segm', 'bbox', 'keypoints']

  annType = annType[1]  # specify type here
  print('Running demo for *%s* results.' % (annType))
  # initialize COCO ground truth api
  cocoGt = COCO(anno_json)
  # initialize COCO detections api
  cocoDt = cocoGt.loadRes(result_json)
  imgIds = sorted(cocoGt.getImgIds())
  imgIds = imgIds[0:100]
  imgIds = imgIds[np.random.randint(100)]
  # running evaluation
  cocoEval = COCOeval(cocoGt, cocoDt, annType)
  cocoEval.params.imgIds = imgIds
  cocoEval.evaluate()
  cocoEval.accumulate()
  cocoEval.summarize()
  pass
