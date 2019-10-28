from __future__ import print_function
import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CUDAExtension

sources = ['roi_align.cc']
headers = ['roi_align.h']
extra_objects = []
defines = []
with_cuda = False

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)

if torch.cuda.is_available():
  print('Including CUDA code.')
  sources += ['roi_align_cuda.cc']
  headers += ['roi_align_cuda.h']
  defines += [('WITH_CUDA', None)]
  with_cuda = True

  extra_objects = ['roi_align_kernel.cu.o']
  extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

cuda_ex = CUDAExtension(
    name='_roi_align',
    sources=sources,
    define_macros=defines,
    extra_objects=extra_objects,
    include_dirs=torch.utils.cpp_extension.include_paths(),
    extra_compile_args={'cxx': ['-g', '--std=c++11'],
                        'nvcc': ['-O2']}
)

if __name__ == '__main__':
  setup(
      name='_roi_align',
      ext_modules=[cuda_ex],
      # inject our custom trigger
      cmdclass={'build_ext': BuildExtension},
  )
