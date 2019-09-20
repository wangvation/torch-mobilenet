#!/usr/bin/python3
# -*- coding:utf-8 -*-

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from utils.metrics import *
import torch.optim as optim
from mobilenet.mobilenet import MobileNetV1
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
          pretrain_model=None):
  transform = transforms.Compose([
      SquarePad(fill="mean"),
      transforms.Resize((256, 256)),
      transforms.RandomCrop(224),
      transforms.RandomHorizontalFlip(0.5),
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
  print("Training model use", torch.cuda.device_count(), "GPUs!")
  if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

  model.to(device)
  if pretrain_model:
    print('Load pretrained weights! ')
    model.load_state_dict(torch.load(pretrain_model, map_location=device))
    pass

  for epoch in range(ephochs):
    test(model, device, loss_fun=loss_fun)
    model.train()
    for step, batch in enumerate(train_loader):
      inputs, labels = batch
      inputs = inputs.to(device)
      labels = labels.to(device)
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = loss_fun(outputs, labels)
      loss.backward()
      optimizer.step()

      # print statistics
      if step % 10 == 0:    # print every 10 mini-batches
        print('[%d, %5d] loss: %.5f' % (epoch + 1, step, loss))

  torch.save(model.state_dict(), pretrain_model)
  print('Training Finished.')


def test(model, device, loss_fun):
  model.eval()

  transform = transforms.Compose([
      SquarePad(fill="mean"),
      transforms.Resize((256, 256)),
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
      val_loss += loss
      _, predicted = torch.max(outputs, 1)
      all_predictions.append(predicted)

      # c = (predicted == labels).squeeze()

      # for i in range(labels.size()[0]):
      #   label = labels[i].item()
      #   class_correct[label] += c[i].item()
      #   class_total[label] += 1

  print('val loss:', val_loss.item() / steps)
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
  correct = torch.sum((tensor_predictions == tensor_labels).squeeze()).item()
  total = tensor_labels.size()[0]
  print('macro  f1 score: %.4f' % mac_f1)
  print('micro  f1 score: %.4f' % mic_f1)
  print('Accuracy  of all : %d / %d , %.2f %%' %
        (correct, total, 100 * correct / total))
  pass


def main(args):
  use_cuda = torch.cuda.is_available()
  cross_entrop = CrossEntropyLoss()
  device = torch.device("cuda" if use_cuda else "cpu")
  print('Use Device:', device)
  model = MobileNetV1(resolution=224, num_classes=120, alpha=1)
  if args.command == "train":
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train(model,
          ephochs=2000,
          optimizer=optimizer,
          device=device,
          loss_fun=cross_entrop,
          pretrain_model=args.model_file)

  elif args.command == "test":
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

  parser.add_argument('-m',
                      '--model-file',
                      default="MobileNet.pt",
                      required=False,
                      metavar="path/to/model-file/",
                      help='Path to model-file')

  args = parser.parse_args()
  main(args)
