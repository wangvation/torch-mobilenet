
import torch
from torchvision import utils
import torchvision.models as models
from matplotlib import pyplot as plt
import numpy as np


def show_save_tensor():

  def vis_tensor(tensor, ch=0, all_kernels=False, nrow=8, padding=2):
    '''
    ch: channel for visualization
    allkernels: all kernels for visualization
    '''
    n, c, h, w = tensor.shape
    if all_kernels:
      tensor = tensor.view(n * c, -1, w, h)
    elif c != 3:
      tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

    # rows = np.min((tensor.shape[0] // nrow + 1, 64))
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    # plt.figure(figsize=(nrow,rows))
    plt.imshow(grid.numpy().transpose((1, 2, 0)))  # CHW HWC

  def save_tensor(tensor, filename, ch=0, all_kernels=False, nrow=8, padding=2):
    n, c, h, w = tensor.shape
    if all_kernels:
      tensor = tensor.view(n * c, -1, w, h)
    elif c != 3:
      tensor = tensor[:, ch, :, :].unsqueeze(dim=1)
    utils.save_image(tensor, filename, nrow=nrow,
                     normalize=True, padding=padding)

  vgg = models.resnet18(pretrained=True)
  mm = vgg.double()
  body_model = [i for i in mm.children()][0]
  # layer1 = body_model[0]
  layer1 = body_model
  tensor = layer1.weight.data.clone()
  vis_tensor(tensor)
  save_tensor(tensor, 'test.png')

  plt.axis('off')
  plt.ioff()
  plt.show()
