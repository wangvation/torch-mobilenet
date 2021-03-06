#!/usr/bin/env bash

# CUDA_PATH=/usr/local/cuda/

export CUDA_PATH=/usr/local/cuda/
#You may also want to ad the following

export CXXFLAGS="-std=c++11"

python3 setup.py build_ext --inplace
rm -rf build

CUDA_ARCH="-gencode arch=compute_30,code=sm_30 \
           -gencode arch=compute_35,code=sm_35 \
           -gencode arch=compute_50,code=sm_50 \
           -gencode arch=compute_52,code=sm_52 \
           -gencode arch=compute_60,code=sm_60 \
           -gencode arch=compute_61,code=sm_61 "

# compile NMS
cd nms/
echo "Compiling nms kernels by nvcc..."
rm -rf build *.o
nvcc -c -o nms_cuda_kernel.cu.o nms_cuda_kernel.cu -x cu -Xcompiler -fPIC $CUDA_ARCH

python3 build.py build_ext --inplace

# compile roi_pooling
cd ../
cd roi_pooling/
echo "Compiling roi pooling kernels by nvcc..."
rm -rf build *.o
nvcc -c -o roi_pooling.cu.o roi_pooling_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
python3 build.py build_ext --inplace

# compile roi_align
cd ../
cd roi_align/
echo "Compiling roi align kernels by nvcc..."
rm -rf build *.o
nvcc -c -o roi_align_kernel.cu.o roi_align_kernel.cu -x cu -Xcompiler -fPIC $CUDA_ARCH
./make.sh

# compile roi_crop
cd ../
cd roi_crop/
rm -rf build *.o
echo "Compiling roi crop kernels by nvcc..."
nvcc -c -o roi_crop_cuda_kernel.cu.o roi_crop_cuda_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
python3 build.py build_ext --inplace

cd ../
