from __future__ import print_function
import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CUDAExtension


sources = ['roi_pooling.cc']
headers = ['roi_pooling.h']
extra_objects = []
defines = []
with_cuda = False

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)

if torch.cuda.is_available():
  print('Including CUDA code.')
  sources += ['roi_pooling_cuda.cc']
  headers += ['roi_pooling_cuda.h']
  defines += [('WITH_CUDA', None)]
  with_cuda = True
  extra_objects = ['roi_pooling.cu.o']
  extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

cuda_ex = CUDAExtension(
    '_roi_pooling',
    sources=sources,
    define_macros=defines,
    include_dirs=torch.utils.cpp_extension.include_paths(),
    extra_objects=extra_objects
)

if __name__ == '__main__':
  setup(
      name='_roi_pooling',
      ext_modules=[cuda_ex],
      # inject our custom trigger
      cmdclass={'build_ext': BuildExtension},
  )
