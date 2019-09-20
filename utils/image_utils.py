#! /usr/bin/python3
# -*- coding:utf-8 -*-
import cv2
import re
# import dlib
import numpy as np

REGEX_FILE_SUFFIX = r'[\.](jpg|png|bmp|jpeg|JPG|PNG|BMP|JPEG)$'
image_regex = re.compile(REGEX_FILE_SUFFIX)


def read_rgb(image_file, height=0, width=0, fx=0., fy=0.):
  bgr_image = cv2.imread(image_file)
  rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
  rgb_image = resize_image(rgb_image, height, width, fx, fy)
  if isinstance(rgb_image, np.ndarray):
    return rgb_image
  else:
    return None


def read_bgr(image_file, height=0, width=0, fx=0., fy=0.):
  bgr_image = cv2.imread(image_file)
  bgr_image = resize_image(bgr_image, height, width, fx, fy)

  if isinstance(bgr_image, np.ndarray):
    return bgr_image
  else:
    return None


def read_gray(image_file, height=0, width=0, fx=0., fy=0.):
  gray_image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
  gray_image = resize_image(gray_image, height, width, fx, fy)
  if isinstance(gray_image, np.ndarray):
    return gray_image
  else:
    return None


def rgb2bgr(rgb_image):
  return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)


def bgr2rgb(bgr_image):
  return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def resize_image(image, height=0, width=0, fx=0., fy=0.):
  old_height, old_width = image.shape[:2]
  if height * width == 0 and height + width != 0:
    height = old_height * width // old_width if height == 0 else height
    width = old_width * height // old_height if width == 0 else width

  if abs(fx * fy) < 1e-6 and abs(fx + fy) > 1e-6:
    fx = fy if -1e-6 < fx < 1e-6 else fx
    fy = fx if -1e-6 < fy < 1e-6 else fy

  if old_height > height > 0 and old_width > width > 0:
    image = cv2.resize(image,
                       (width, height),
                       interpolation=cv2.INTER_AREA)

  elif height > old_height and width > old_width:
    image = cv2.resize(image,
                       (width, height),
                       interpolation=cv2.INTER_CUBIC)
  elif height > 0 and width > 0:
    image = cv2.resize(image,
                       (width, height),
                       interpolation=cv2.INTER_LINEAR)

  if 1.0 > fx > 0.0 and 1.0 > fy > 0.0:
    image = cv2.resize(image,
                       None,
                       fx=fx,
                       fy=fy,
                       interpolation=cv2.INTER_AREA)

  elif fx > 1.0 and fy > 1.0:
    image = cv2.resize(image,
                       None,
                       fx=fx,
                       fy=fy,
                       interpolation=cv2.INTER_CUBIC)
  elif fx > 0.0 and fy > 0.0:
    image = cv2.resize(image,
                       None,
                       fx=fx,
                       fy=fy,
                       interpolation=cv2.INTER_LINEAR)

  return image


def write_rgb(rgb_image, image_file, height=0, width=0, fx=0., fy=0.):
  rgb_image = resize_image(rgb_image, height, width, fx, fy)
  bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
  cv2.imwrite(image_file, bgr_image)


def write_gray(gray_image, image_file, height=0, width=0, fx=0., fy=0.):
  gray_image = resize_image(gray_image, height, width, fx, fy)
  cv2.imwrite(image_file, gray_image)


def write_bgr(image, image_file, height=0, width=0, fx=0., fy=0.):
  image = resize_image(image, height, width, fx, fy)
  cv2.imwrite(image_file, image)


def png2jpg(png_file, jpg_file):
  if not png_file.endswith('.png'):
    raise ValueError('png_file must be ends with .png!')

  if not jpg_file.endswith('.jpg'):
    raise ValueError('jpg_file must be ends with .jpg!')

  png_image = cv2.imread(png_file, cv2.IMREAD_UNCHANGED)
  bgr_image = cv2.cvtColor(png_image, cv2.COLOR_BGRA2BGR)
  cv2.imwrite(jpg_file, bgr_image)


def draw_rect(bgr_image, rect, color):
  height, width = bgr_image.shape[:2]
  if isinstance(rect, (list, tuple)):
    left, top, right, bottom = rect
  else:
    left = rect.left()
    top = rect.top()
    right = rect.right()
    bottom = rect.bottom()
  cv2.rectangle(bgr_image,
                (left, top),
                (right, bottom),
                color=color,
                thickness=max(round(3 * (height / 1920)), 1))


def draw_bbox(bgr_image, x, y, w, h, color):
  draw_rect(bgr_image, (x, y, x + w, y + h), color)


def draw_text(bgr_image, text, x, y, color):
  height, width = bgr_image.shape[:2]
  thickness = max(round(3 * (height / 1920)), 1)
  cv2.putText(bgr_image,
              text,
              (int(x), int(y)),
              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
              fontScale=max(1.2, 1.2 * (height / 1920)),
              thickness=thickness,
              color=color,
              lineType=2)
  pass


def padding_image(image, dst_height, dst_width, pad_value=0):

  height, width, channel = image.shape
  if dst_height == height and dst_width == width:
    return image
  pad_image = np.full(shape=(dst_height, dst_width, channel),
                      fill_value=pad_value,
                      dtype=image.dtype)
  pad_image[0:height, 0:width] = image
  return pad_image


def apply_mask(image, mask, foreground, color, alpha=1):
  """
  Apply the given mask to the image.
  """
  for c in range(3):
    image[:, :, c] = np.where(mask == foreground,
                              image[:, :, c] * (1 - alpha) + alpha * color[c],
                              image[:, :, c])
  return image


def crop_image(image, rect):
  """
  """
  height, width = image.shape[:2]
  left, top, right, bottom = rect
  left = max(0, left)
  top = max(0, top)
  right = min(right, width - 1)
  bottom = min(bottom, height - 1)
  cropped_image = image[top:bottom + 1, left:right + 1, ...]
  return cropped_image


def paste_image(image, paste, rect):
  """
  """
  height, width = image.shape[:2]
  left, top, right, bottom = rect
  left = max(0, left)
  top = max(0, top)
  right = min(right, width - 1)
  bottom = min(bottom, height - 1)
  paste_height = bottom - top
  paste_width = right - left
  image[top:bottom + 1, left:right + 1, ...] = paste[0:paste_height + 1,
                                                     0:paste_width + 1,
                                                     ...]
  return image


def draw_points(image,
                points_pts,
                radius=3,
                color=(0, 0, 255),
                fontScale=1.2,
                thickness=3,
                point_size=3,
                with_number=False):
  '''
    draw points on image.
  '''
  for (x, y) in points_pts:
    cv2.circle(image, (int(x), int(y)), point_size, color, thickness=thickness)

  if with_number:
    for i, point in enumerate(points_pts):
      x, y = point
      cv2.putText(image,
                  str(i),
                  (int(x), int(y) - 10),
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                  fontScale=fontScale,
                  thickness=thickness,
                  color=color,
                  lineType=2)
  return image
