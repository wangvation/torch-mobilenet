from __future__ import print_function
import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CUDAExtension

sources = ['roi_crop.cc']
headers = ['roi_crop.h']
defines = []
with_cuda = False

if torch.cuda.is_available():
  print('Including CUDA code.')
  sources += ['roi_crop_cuda.cc']
  headers += ['roi_crop_cuda.h']
  defines += [('WITH_CUDA', None)]
  with_cuda = True

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
extra_objects = ['roi_crop_cuda_kernel.cu.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

cuda_ex = CUDAExtension(
    '_roi_crop',
    sources=sources,
    define_macros=defines,
    extra_objects=extra_objects,
    extra_compile_args={'cxx': ['-Wno-cpp',
                                '-Wno-unused-function'],
                        'nvcc': ['-O2']}
)

if __name__ == '__main__':
  setup(
      name='_roi_crop',
      ext_modules=[cuda_ex],
      # inject our custom trigger
      cmdclass={'build_ext': BuildExtension},
  )
