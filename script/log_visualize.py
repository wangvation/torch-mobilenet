import matplotlib.pyplot as plt
import os
import json


def log_analysis(log_file):
  with open(log_file, 'w') as lf:
    lines = lf.readlines()

  objs = []
  for line in lines:
    obj = json.loads(line.strip())
    objs.append(obj)

  return objs


def main():
  pass
