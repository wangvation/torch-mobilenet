#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os
import shutil
import argparse
from tqdm import tqdm


def rename_anno_file(path):
  for root, dirs, files in os.walk(path, followlinks=True):
    for file in tqdm(files):
      if not file.endswith('.xml'):
        prefix, ext = os.path.splitext(file)
        anno_file = os.path.join(root, file)
        xml_file = os.path.join(root, prefix + '.xml')
        shutil.move(anno_file, xml_file)


def main(agrs):
  rename_anno_file(agrs.xml_dir)
  pass


if __name__ == '__main__':
  parser = argparse.ArgumentParser('rename files to xml in batches.')
  parser.add_argument('-x',
                      '--xml-dir',
                      type=str,
                      help='path to xml file dir.')
  args = parser.parse_args()
  main(args)
