"""MS COCO Detection Evaluate Metrics."""
from __future__ import absolute_import

import sys
import json
import torch
try:
  from StringIO import StringIO
except ImportError:
  from io import StringIO
import os
import warnings
import numpy as np
from pycocotools.cocoeval import COCOeval


class COCODetectionMetric(object):
  """Detection metric for COCO bbox task.

  Parameters
  ----------
  dataset : instance of pycocotools.coco.COCO
    The validation dataset.
  save_prefix : str
    Prefix for the saved JSON results.
  use_time : bool
    Append unique datetime string to created JSON file name if ``True``.
  cleanup : bool
    Remove created JSON file if ``True``.
  score_thresh : float
    Detection results with confident scores smaller than ``score_thresh`` will
    be discarded before saving to results.

  """

  def __init__(self,
               dataset,
               save_prefix,
               cleanup=False,
               score_thresh=0.05):
    super(COCODetectionMetric, self).__init__()
    self.dataset = dataset
    self._img_ids = sorted(dataset.coco.getImgIds())
    self._current_id = 0
    self._cleanup = cleanup
    self._results = []
    self._score_thresh = score_thresh

  def __del__(self):
    if self._cleanup:
      try:
        os.remove(self._filename)
      except IOError as err:
        warnings.warn(str(err))

  def reset(self):
    self._current_id = 0
    self._results = []

  def _update(self):
    """Use coco to get real scores. """
    if not self._current_id == len(self._img_ids):
      warnings.warn(
          'Recorded {} out of {} validation images, incomplete results'.format(
              self._current_id, len(self._img_ids)))
    if not self._results:
      # in case of empty results, push a dummy result
      self._results.append({'image_id': self._img_ids[0],
                            'category_id': 0,
                            'bbox': [0, 0, 0, 0],
                            'score': 0})
    try:
      with open(self._filename, 'w') as f:
        json.dump(self._results, f)
    except IOError as e:
      raise RuntimeError(
          "Unable to dump json file, ignored. What(): {}".format(str(e)))

    pred = self.dataset.coco.loadRes(self._filename)
    gt = self.dataset.coco
    coco_eval = COCOeval(gt, pred, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    self._coco_eval = coco_eval
    return coco_eval

  def get(self):
    """Get evaluation metrics. """
    # Metric printing adapted from detectron/json_dataset_evaluator.
    def _get_thr_ind(coco_eval, thr):
      ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                     (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
      iou_thr = coco_eval.params.iouThrs[ind]
      assert np.isclose(iou_thr, thr)
      return ind

    # call real update
    try:
      coco_eval = self._update()
    except IndexError:
      # invalid model may result in empty JSON results, skip it
      return ['mAP', ], ['0.0', ]

    IoU_lo_thresh = 0.5
    IoU_hi_thresh = 0.95
    ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
    ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
    # precision has dims (iou, recall, cls, area range, max dets)
    # area range index 0: all area ranges
    # max dets index 2: 100 per image
    precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
    ap_default = np.mean(precision[precision > -1])
    names, values = [], []
    names.append('~~~~ Summary metrics ~~~~\n')
    # catch coco print string, don't want directly print here
    _stdout = sys.stdout
    sys.stdout = StringIO()
    coco_eval.summarize()
    coco_summary = sys.stdout.getvalue()
    sys.stdout = _stdout
    values.append(str(coco_summary).strip())
    for cls_ind, cls_name in enumerate(self.dataset.classes):
      precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1),
                                              :,
                                              cls_ind, 0, 2]
      ap = np.mean(precision[precision > -1])
      names.append(cls_name)
      values.append('{:.1f}'.format(100 * ap))
    # put mean AP at last, for comparing perf
    names.append('~~~~ MeanAP @ IoU=[{:.2f},{:.2f}] ~~~~\n'.format(
        IoU_lo_thresh, IoU_hi_thresh))
    values.append('{:.1f}'.format(100 * ap_default))
    return names, values

  # pylint: disable=arguments-differ, unused-argument
  def update(self, pred_bboxes, pred_labels, pred_scores, *args, **kwargs):
    """Update internal buffer with latest predictions.
    Note that the statistics are not available until you call self.get()
    to return the metrics.

    Parameters
    ----------
    pred_bboxes : torch.Tensor or numpy.ndarray
        Prediction bounding boxes with shape `B, N, 4`.
        Where B is the size of mini-batch, N is the number of bboxes.
    pred_labels : torch.Tensor or numpy.ndarray
        Prediction bounding boxes labels with shape `B, N`.
    pred_scores : torch.Tensor or numpy.ndarray
        Prediction bounding boxes scores with shape `B, N`.

    """
    def as_numpy(a):
      """Convert a (list of) torch.Tensor into numpy.ndarray"""
      if isinstance(a, (list, tuple)):
        out = [x.asnumpy() if isinstance(x, torch.Tensor) else x for x in a]
        return np.concatenate(out, axis=0)
      elif isinstance(a, mx.nd.NDArray):
        a = a.asnumpy()
      return a

    for pred_bbox, pred_label, pred_score in zip(
            *[as_numpy(x) for x in [pred_bboxes, pred_labels, pred_scores]]):
      valid_pred = np.where(pred_label.flat >= 0)[0]
      pred_bbox = pred_bbox[valid_pred, :].astype(np.float)
      pred_label = pred_label.flat[valid_pred].astype(int)
      pred_score = pred_score.flat[valid_pred].astype(np.float)

      imgid = self._img_ids[self._current_id]
      self._current_id += 1
      height_scale, width_scale = (1., 1.)
      # for each bbox detection in each image
      for bbox, label, score in zip(pred_bbox, pred_label, pred_score):
        if label not in self.dataset.contiguous_id_to_json:
          # ignore non-exist class
          continue
        if score < self._score_thresh:
          continue
        category_id = self.dataset.contiguous_id_to_json[label]
        # rescale bboxes
        bbox[[0, 2]] *= width_scale
        bbox[[1, 3]] *= height_scale
        # convert [xmin, ymin, xmax, ymax]  to [xmin, ymin, w, h]
        bbox[2:4] -= (bbox[:2] - 1)
        self._results.append({'image_id': imgid,
                              'category_id': category_id,
                              'bbox': bbox[:4].tolist(),
                              'score': score})
