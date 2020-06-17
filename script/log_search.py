#! /usr/bin/python3
# -*- coding:utf8 -*-

import os
import argparse
import re

# debug for test.png
# debug for test.jpeg
# debug for test.PNG
# debug for test.JPG

def search_log_file(log_file, pattern_str, regex_str=None):

  if regex_str is not None:
    print(regex_str)
    regex = re.compile(regex_str)

  with open(log_file, 'r') as log_f:
    lines = log_f.readlines()
  lines = [line.strip() for line in lines]

  for line in lines:
    if regex_str is None and pattern_str in line:
      print(line)
      # do something ...
    elif regex_str is not None and regex.search(line) is not None:
      print(line)
      # do something ...


def main(args):
  if args.pattern_str is None and args.regex_str is None:
    print('--pattern-str 与 --regex-str 不能同时为空')
    return

  if not os.path.exists(args.log_file):
    print(args.log_file, ' 文件不存在')
    return
  search_log_file(args.log_file, args.pattern_str, args.regex_str)
  pass


def __parse_args():
  parser = argparse.ArgumentParser("Log file searcher tool.")

  parser.add_argument("-l",
                      "--log-file",
                      required=True,
                      type=str,
                      help="path/to/log-file")

  parser.add_argument("-p",
                      "--pattern-str",
                      required=False,
                      type=str,
                      help="search log file by pattern-str")

  parser.add_argument("-r",
                      "--regex-str",
                      required=False,
                      type=str,
                      help="search log file by regex-str")

  return parser.parse_args()


if __name__ == '__main__':
  args = __parse_args()
  main(args)
