
import os
from torchvision.datasets import VisionDataset
from torchvision import transforms
F = transforms.functional


class StanfordDogsDataset(VisionDataset):
  """docstring for StanfordDogsDataset"""

  def __init__(self,
               root,
               data_list_file=None,
               train=True,
               transform=None,
               target_transform=None,
               parse_class_name=None):
    """



    Arguments:
      root {[path]} -- dataset root directory
      root is a path-like object,the directory tree example:
        root/
          Annotation/
              class_name1/
              file1,file2,...
            class_name2/
              file1,file2,...
            ...
          Images/
            class_name1/
              file1,file2,...
            class_name2/
              file1,file2,...
            ...

      data_list_file {[path]} -- data list file.
      data_list_file is a file-like object, the file context example:
        image_file,class
        image_name1,label1,
        image_name2,label2
    """
    super(StanfordDogsDataset, self).__init__(
        root=root, transform=transform, target_transform=target_transform)

    self.parse_class_name = parse_class_name
    self.image_loader = None
    self.image_folder = os.path.join(self.root, "Images")

    try:
      # read image by PIL
      from PIL import Image
      self.image_loader = Image.open

    except ImportError:
      pass

    if self.image_loader is None:
      try:
        # read image by opencv
        from image_utils import read_rgb
        self.image_loader = read_rgb
      except ImportError:
        from PIL import Image
        self.image_loader = Image.open

    if self.image_loader is None:
      try:
        # read image by skimage
        from skimage import io
        self.image_loader = io.imread

      except ImportError:
        pass

    if self.image_loader is None:
      raise ImportError("Can't import OpenCV 3.x, PIL or scikit-image,"
                        " please install one of them.")

    if data_list_file:
      data_list_file = os.path.join(root, data_list_file)
      self.file_files, self.targets = self.__load_from_file(data_list_file)
    else:
      self.file_files, self.targets = self.__load_from_dir(self.image_folder)

    self.classes = self.__make_classes(self.file_files, self.targets)

  def __make_classes(self, file_files, targets):
    classes = {}
    for i in range(len(self)):
      img_file, target = file_files[i], int(targets[i])
      class_name, target = self.__make_label_name_pair(img_file, target)
      classes[target] = class_name
    return classes

  def __make_label_name_pair(self, class_name, target):

    if self.target_transform is not None:
      target = self.target_transform(target)

    if self.parse_class_name is not None:
      class_name = self.parse_class_name(class_name)

    return class_name, target

  def __load_from_file(self, file):
    with open(file, 'r') as f:
      lines = f.readlines()

    file_files = []
    targets = []
    for line in lines[1:]:  # skip first line.
      file, label = line.split(',')
      file_files.append(os.path.join(self.image_folder, file.strip()))
      targets.append(label.strip())
    return file_files, targets

  def __load_from_dir(self, path):
    file_files = []
    targets = []
    for i, _dir in enumerate(os.listdir(path)):
      for file in os.listdir(path):
        file_files.append(os.path.join(path, _dir, file))
        targets.append(i)

    return file_files, targets

  def __len__(self):
    return len(self.file_files)

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img_file, target = self.file_files[index], self.targets[index]

    # doing this so that it is consistent with all other datasets
    # to return a PIL Image

    img = self.image_loader(img_file)
    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target


class SquarePad(object):
  def __init__(self, fill="zero", padding_mode='constant'):
    if fill == "zero":
      self.fill = 0
    elif fill == "mean":
      self.fill = (124, 116, 104)

    self.padding_mode = padding_mode

  def __call__(self, img):
    """
    Args:
        img (PIL Image): Image to be padded.

    Returns:
        PIL Image: Padded image.
    """
    w, h = img.size
    max_edge = max(w, h)
    padding = ((max_edge - w) // 2, (max_edge - h) // 2)
    return F.pad(img, padding, self.fill, self.padding_mode)

  def __repr__(self):
    return self.__class__.__name__ + \
        '( fill={1}, padding_mode={2})'.format(self.fill, self.padding_mode)
  pass
