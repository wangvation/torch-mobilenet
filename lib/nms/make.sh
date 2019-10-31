#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

CUDA_ARCH="-gencode arch=compute_30,code=sm_30 \
           -gencode arch=compute_35,code=sm_35 \
           -gencode arch=compute_50,code=sm_50 \
           -gencode arch=compute_52,code=sm_52 \
           -gencode arch=compute_60,code=sm_60 \
           -gencode arch=compute_61,code=sm_61 "


echo "Compiling stnm kernels by nvcc..."

rm -rf build *.o

nvcc -c -o nms_cuda_kernel.cu.o nms_cuda_kernel.cu -x cu -Xcompiler -fPIC $CUDA_ARCH

python3 build.py build_ext --inplace
