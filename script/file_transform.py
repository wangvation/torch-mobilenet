#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os
import argparse
from scipy import io as sio
from tqdm import tqdm


def mat_file_transfrom(path):
  for root, dirs, files in os.walk(path, followlinks=True):
    for file in tqdm(files):
      if file.endswith('.mat'):
        prefix, ext = os.path.splitext(file)
        mat_file = os.path.join(root, file)
        mat_dict = sio.loadmat(mat_file)
        file_list = mat_dict['file_list']
        labels = mat_dict['labels']
        txt_file = os.path.join(root, prefix + '.txt')
        txt_list = []
        txt_list.append('image_file,label')
        for image_file, label in zip(file_list, labels):
          txt_list.append('%s,%s' % (image_file[0][0], label[0]))

        with open(txt_file, 'w') as tf:
          tf.writelines('\n'.join(txt_list))
          tf.write('\n')


def main(agrs):
  mat_file_transfrom(agrs.input_dir)
  pass


if __name__ == '__main__':
  parser = argparse.ArgumentParser('covert matlab file to text file.')
  parser.add_argument('-i',
                      '--input-dir',
                      type=str,
                      help='path to input dir.')
  args = parser.parse_args()
  main(args)
