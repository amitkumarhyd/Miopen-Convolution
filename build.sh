#!/bin/bash
source setup.sh

if ![ -d "$1" ]; then
  mkdir $1
  echo "Created directory: $1"
fi
cd $1

export CC=clang
export CXX=clang++

cmake -DCMAKE_BUILD_TYPE=Debug -DDNNL_CPU_RUNTIME=OMP -DDNNL_GPU_RUNTIME=DPCPP -DDNNL_GPU_VENDOR=AMD .. \
&& make -j26 |& tee log

#cmake -DCMAKE_BUILD_TYPE=Debug -DDNNL_CPU_RUNTIME=DPCPP -DDNNL_GPU_RUNTIME=DPCPP -DDNNL_GPU_VENDOR=AMD .. \
#&& make -j26 |& tee log
