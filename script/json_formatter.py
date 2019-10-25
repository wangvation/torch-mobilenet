import json
import os
import argparse

from tqdm import tqdm


def json_format(path, indent=2):
  for root, dirs, files in os.walk(path, followlinks=True):
    for file in tqdm(files):
      if file.endswith('.json'):

        json_file = os.path.join(root, file)
        with open(json_file, 'r') as jf:
          json_obj = json.load(jf)

        # print(type(json_obj))
        with open(json_file, 'w') as jf:
          json.dump(json_obj, jf, indent=indent)


def main(agrs):
  json_format(agrs.input_dir, indent=2)
  pass


if __name__ == '__main__':
  parser = argparse.ArgumentParser('convert json file to readable.')
  parser.add_argument('-i',
                      '--input-dir',
                      type=str,
                      help='path to input dir.')
  args = parser.parse_args()
  main(args)
