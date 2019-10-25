from __future__ import print_function
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

extra_objects = ['nms_cuda_kernel.cu.o']
print(extra_objects)

cuda_ex = CUDAExtension(
    '_nms',
    sources=sources,
    define_macros=defines,
    extra_objects=extra_objects,
    include_dirs=torch.utils.cpp_extension.include_paths(),
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
