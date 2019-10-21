#!/usr/bin/python3
# -*- coding:utf-8 -*-

import torch
import dlib
# import numpy as np


def precision(prediction, labels, class_num):
  """precision for each classes"""
  _, predicted = torch.max(prediction, 1)
  precisions = []
  for i in range(class_num):
    pred = (predicted == i).squeeze()
    gt = (labels == i).squeeze()
    tp = torch.sum((pred & gt).squeeze()).item()
    fp = torch.sum(pred).item() - tp
    precisions.append(tp / (tp + fp))
  return precisions


def recall(prediction, labels, class_num):
  """recall for each classes"""
  _, predicted = torch.max(prediction, 1)
  recalles = []
  for i in range(class_num):
    pred = (predicted == i).squeeze()
    gt = (labels == i).squeeze()
    tp = torch.sum((pred & gt).squeeze()).item()
    fn = torch.sum(gt).item() - tp
    recalles.append(tp / (tp + fn))
  return recalles


def accuracy(prediction, labels, class_num):
  """accuracy for each classes"""
  _, predicted = torch.max(prediction, 1)
  accuracys = []
  for i in range(class_num):
    pred = (predicted == i).squeeze()
    gt = (labels == i).squeeze()
    tp = torch.sum((pred & gt).squeeze()).item()
    total = torch.sum(gt).item()
    accuracys.append(tp / total)
  return accuracys


def all_accuracy(prediction, labels):
  _, predicted = torch.max(prediction, 1)
  correct = torch.sum((predicted == labels).squeeze()).item()
  total = labels.size()[0]
  return correct, total, correct / total


def f1_score(prediction, labels, class_num):
  """f1_score for each classes"""
  _, predicted = torch.max(prediction, 1)
  f1_scores = []
  for i in range(class_num):
    pred = (predicted == i).squeeze()
    gt = (labels == i).squeeze()
    tp = torch.sum((pred & gt).squeeze()).item()
    fp = torch.sum(pred).item() - tp
    fn = torch.sum(gt).item() - tp
    f1_scores.append(2 * tp / (2 * tp + fp + fn))
  return f1_scores


def macro_f1(prediction, labels, class_num):
  """macro_f1 for each classes"""
  f1_scores = f1_score(prediction, labels, class_num)
  return sum(f1_scores) / class_num


def micro_f1(prediction, labels, class_num):
  """micro_f1 for each classes"""
  _, predicted = torch.max(prediction, 1)
  tps = []
  fps = []
  fns = []
  for i in range(class_num):
    pred = (predicted == i).squeeze()
    gt = (labels == i).squeeze()
    tp = torch.sum((pred & gt).squeeze()).item()
    fp = torch.sum(pred).item() - tp
    fn = torch.sum(gt).item() - tp
    tps.append(tp)
    fps.append(fp)
    fns.append(fn)

  mic_presision = sum(tps) / (sum(tps) + sum(fps))
  mic_recall = sum(tps) / (sum(tps) + sum(fns))
  micro_f1 = 2 * (mic_presision * mic_recall) / (mic_presision + mic_recall)
  return micro_f1


def mean_averge_precision(prediction, labels):
  _, predicted = torch.max(prediction, 1)
  pass


def pascal_voc_map(prediction, labels):
  _, predicted = torch.max(prediction, 1)
  pass


def roc(prediction, labels, thresholds=None):
  """
  TPR=TP/(TP+FN)
  FPR=FP/(FP+TN)
  """
  if thresholds is None:
    thresholds = [x / 100 for x in range(0, 101, 10)]
  scores, predicted = torch.max(prediction, 1)

  for thres in thresholds:
    pass
  pass


def auc(prediction, labels, thresholds=None):
  roces = roc(prediction, labels)
  """
  ROC曲线下的面积
  Area Under Curve
  """
  area = 0.0
  for r in roces:
    delta_thres = 0.1
    area += r * delta_thres
  return area


def iou(rect1, rect2):
  if isinstance(rect1, dlib.rectangle):
    left1, top1 = rect1.left(), rect1.top()
    right1, bottom1 = rect1.right(), rect1.bottom()
  elif isinstance(rect1, (list, tuple)):
    left1, top1, right1, bottom1 = rect1

  if isinstance(rect2, dlib.rectangle):
    left2, top2 = rect2.left(), rect2.top()
    right2, bottom2 = rect2.right(), rect2.bottom()
  elif isinstance(rect2, (list, tuple)):
    left2, top2, right2, bottom2 = rect2

  cross_w = min(right1, right2) - max(left1, left2)
  corss_h = min(bottom1, bottom2) - max(top1, top2)
  if cross_w <= 0 or corss_h <= 0:
    return 0
  area1 = (right1 - left1) * (bottom1 - top1)
  area2 = (right2 - left2) * (bottom2 - top2)
  intersection = cross_w * corss_h
  union = area1 + area2 - intersection
  return intersection / union
