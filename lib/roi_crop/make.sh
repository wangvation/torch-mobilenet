#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

echo "Compiling my_lib kernels by nvcc..."
nvcc -c -o roi_crop_cuda_kernel.cu.o roi_crop_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

python3 build.py build_ext --inplace
