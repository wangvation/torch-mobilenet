#!/usr/bin/env bash

export CUDA_PATH=/usr/local/cuda/

echo "Compiling roi_align kernels by nvcc..."
nvcc -c -o roi_align_kernel.cu.o roi_align_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

python3 build.py build_ext --inplace
