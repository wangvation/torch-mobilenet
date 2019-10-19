#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from utils.metrics import *
import torch.optim as optim
from module.mobilenet import MobileNetV1
from module.mobilenet import MobileNetV2
from dataset.stanford_dogs import StanfordDogsDataset
from dataset.stanford_dogs import SquarePad
# import numpy as np
import argparse


def __parse_class_name(file):
  folder_name = file.split('/')[-2]
  class_name = folder_name.split('-')[-1]
  return class_name


def train(model,
          ephochs,
          optimizer,
          device,
          loss_fun,
          model_weights=None,
          model_dir=None,
          log_dir=None):

  transform = transforms.Compose([
      SquarePad(fill="mean"),
      transforms.Resize((229, 229)),
      transforms.RandomCrop(224),
      transforms.RandomHorizontalFlip(0.5),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.483, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
  ])
  val_transform = transforms.Compose([
      SquarePad(fill="mean"),
      transforms.Resize((229, 229)),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.483, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
  ])
  train_set = StanfordDogsDataset(root="./data/stanford_dogs",
                                  data_list_file="train_list.txt",
                                  transform=transform,
                                  target_transform=lambda x: int(x) - 1,
                                  parse_class_name=__parse_class_name)
  train_loader = torch.utils.data.DataLoader(train_set,
                                             batch_size=64,
                                             shuffle=True)

  val_set = StanfordDogsDataset(root="./data/stanford_dogs",
                                data_list_file="test_list.txt",
                                transform=val_transform,
                                target_transform=lambda x: int(x) - 1,
                                parse_class_name=__parse_class_name)

  val_loader = torch.utils.data.DataLoader(val_set,
                                           batch_size=64,
                                           shuffle=True)
  print("Training model use", torch.cuda.device_count(), "GPUs!")
  if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

  model.to(device)
  model_fmt = "%s_epoch:%04d.pt"
  start_epoch, end_epoch = ephochs
  if model_weights and os.path.exists(model_weights):
    print('Load pretrained model: %s' % model_weights)
    model.load_state_dict(torch.load(model_weights, map_location=device))
    pass

  log_file = open(os.path.join(log_dir, "%s_training.log" % model.name), 'w')
  for epoch in range(start_epoch, end_epoch):
    test(model,
         device,
         loss_fun=loss_fun,
         val_loader=val_loader,
         log_file=log_file)

    sum_loss = 0.0
    for step, batch in enumerate(train_loader):
      inputs, labels = batch
      inputs = inputs.to(device)
      labels = labels.to(device)
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = loss_fun(outputs, labels)
      loss.backward()
      optimizer.step()
      sum_loss += loss.item()
      # print statistics
      if step % 10 == 0:    # print every 10 mini-batches
        print('[%d, %5d] loss: %.5f' % (epoch + 1, step, loss.item()))
        if log_file:
          print('[%d, %5d] loss: %.5f' %
                (epoch + 1, step, loss.item()), file=log_file)

    print('%d ephochs train loss: %.5f' % (epoch + 1, sum_loss / step))
    model_file = os.path.join(model_dir, model_fmt % (model.name, epoch + 1))
    torch.save(model.state_dict(), model_file)
  log_file.close()
  print('Training Finished.')


def test(model, device, loss_fun, val_loader=None, log_file=None):

  if not val_loader:
    transform = transforms.Compose([
        SquarePad(fill="mean"),
        transforms.Resize((229, 229)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.483, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_set = StanfordDogsDataset(root="./data/stanford_dogs",
                                  data_list_file="test_list.txt",
                                  transform=transform,
                                  target_transform=lambda x: int(x) - 1,
                                  parse_class_name=__parse_class_name)

    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=64,
                                             shuffle=True)
  all_predictions = []
  all_labels = []
  val_loss = 0
  steps = 0
  with torch.no_grad():
    for data in val_loader:
      steps += 1
      images, labels = data
      all_labels.append(labels)
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)
      loss = loss_fun(outputs, labels)
      val_loss += loss.item()
      all_predictions.append(outputs)

  print('val loss:', val_loss / steps)
  tensor_predictions = torch.cat(all_predictions, dim=0)
  tensor_labels = torch.cat(all_labels, dim=0)
  tensor_predictions = tensor_predictions.to(torch.device('cpu'))
  tensor_labels = tensor_labels.to(torch.device('cpu'))
  # accuracys = accuracy(tensor_predictions, tensor_labels, class_num=10)
  # precisions = precision(tensor_predictions, tensor_labels, class_num=10)
  # recalles = recall(tensor_predictions, tensor_labels, class_num=10)
  # f1_scores = f1_score(tensor_predictions, tensor_labels, class_num=10)
  # for i in range(10):
  #   print('metrics of %2s (precision:%.4f, recall:%.4f'
  #         ', f1_score:%.4f, accuracy:%.4f) ' % (classes[i], precisions[i],
  #                                               recalles[i], f1_scores[i],
  #                                               accuracys[i]))
  mac_f1 = macro_f1(tensor_predictions, tensor_labels, class_num=120)
  mic_f1 = micro_f1(tensor_predictions, tensor_labels, class_num=120)
  correct, total, acc = all_accuracy(tensor_predictions, tensor_labels)
  print('macro  f1 score: %.4f' % mac_f1)
  print('micro  f1 score: %.4f' % mic_f1)
  print('Accuracy  of all : %d / %d , %.2f %%' %
        (correct, total, 100 * acc))

  if log_file:
    print('macro  f1 score: %.4f' % mac_f1, file=log_file)
    print('micro  f1 score: %.4f' % mic_f1, file=log_file)
    print('Accuracy  of all : %d / %d , %.2f %%' %
          (correct, total, 100 * acc), file=log_file)
  pass


def main(args):
  use_cuda = torch.cuda.is_available()
  cross_entrop = CrossEntropyLoss()
  device = torch.device("cuda" if use_cuda else "cpu")
  print('Use Device:', device)
  # model = MobileNetV1(resolution=224, num_classes=120, alpha=1)
  model = MobileNetV2(resolution=224, num_classes=120, multiplier=1)
  if args.command == "train":
    model.train()
    optimizer = optim.SGD(model.parameters(),
                          lr=0.001,
                          momentum=0.9,
                          weight_decay=1e-5,
                          nesterov=False)
    train(model,
          ephochs=[303, 1000],
          optimizer=optimizer,
          device=device,
          loss_fun=cross_entrop,
          model_weights=args.weights,
          model_dir=args.model_dir,
          log_dir=args.log_dir)

  elif args.command == "test":
    model.eval()
    model.load_state_dict(torch.load(args.model_file, map_location=device))
    test(model, device, loss_fun=cross_entrop)
  pass


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='MobileNet Classfier')

  parser.add_argument('command',
                      default="train",
                      choices=['train', 'test'],
                      help='train or test')

  parser.add_argument('-i',
                      '--data-dir',
                      required=False,
                      metavar="path/to/data-dir/",
                      help='Path to data-dir')

  parser.add_argument('-w',
                      '--weights',
                      required=False,
                      metavar="path/to/weights/",
                      help='Path to weights')

  parser.add_argument('-m',
                      '--model-dir',
                      default="checkpoint",
                      required=False,
                      metavar="path/to/model-dir/",
                      help='Path to model-dir')

  parser.add_argument('-l',
                      '--log-dir',
                      default="log",
                      required=False,
                      metavar="path/to/log-dir/",
                      help='Path to log-dir')

  args = parser.parse_args()
  main(args)
