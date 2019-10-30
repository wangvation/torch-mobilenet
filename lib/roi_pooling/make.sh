#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

echo "Compiling roi pooling kernels by nvcc..."
nvcc -c -o roi_pooling.cu.o roi_pooling_kernel.cu \
   -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
python3 build.py build_ext --inplace

rm -rf build *.o