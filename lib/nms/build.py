from __future__ import print_function
import os
import torch

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CUDAExtension

# this_file = os.path.dirname(__file__)

sources = []
headers = []
defines = []
with_cuda = False

if torch.cuda.is_available():
  print('Including CUDA code.')
  sources += ['nms_cuda.cc']
  headers += ['nms_cuda.h']
  defines += [('WITH_CUDA', None)]
  with_cuda = True

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
extra_objects = ['nms_cuda_kernel.cu.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]
print(extra_objects)

cuda_ex = CUDAExtension(
    '_nms',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects,
    extra_compile_args={'cxx': ['-Wno-cpp',
                                '-Wno-unused-function',
                                '-std=c++11'],
                        'nvcc': ['-O2']}
)

if __name__ == '__main__':
  setup(
      name='_nms',
      ext_modules=[cuda_ex],
      # inject our custom trigger
      cmdclass={'build_ext': BuildExtension},
  )
