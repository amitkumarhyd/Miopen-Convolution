#!/bin/bash
export SYCL_DEVICE_FILTER=hip

# 5.1.3
ROCM_ROOT=/opt/rocm

# From MIOpen from rocm package
#export LD_LIBRARY_PATH=${ROCM_ROOT}miopen/lib:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=${ROCM_ROOT}/hip/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${ROCM_ROOT}/rocblas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${ROCM_ROOT}/lib/:$LD_LIBRARY_PATH

# Custom MIOpen
#export MIOPENROOT=/opt/ms1_user/DNN/MIOpen/build_2
#export MIOPENROOT=/opt/ms1_user/DNN/MIOpen/build_4
export MIOPENROOT=/opt/ms1_user/DNN/MIOpen/build_5
#export MIOPENROOT=/opt/ms1_user/DNN/MIOpen/build_6
#export MIOPENROOT=${ROCM_ROOT}
export LD_LIBRARY_PATH=${MIOPENROOT}/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=${MIOPENROOT}/install/lib:$LD_LIBRARY_PATH
export MIOPEN_DISABLE_CACHE=1

source /opt/intel/oneapi/tbb/latest/env/vars.sh
CMPLRROOT=/opt/compiler/llvm/build
#CMPLRROOT=/opt/ms1_user/compiler/llvm/build
#CMPLRROOT=/opt/compiler/llvm/build_3

# For LLD
export PATH=${CMPLRROOT}/bin:$PATH
export PATH=${CMPLRROOT}/install/bin:$PATH
export LD_LIBRARY_PATH=${CMPLRROOT}/install/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=${CMPLRROOT}/install/lib/clang/15.0.0/include:$LD_LIBRARY_PATH
export LIBRARY_PATH=${CMPLRROOT}/install/lib:$LIBRARY_PATH
export CPATH=${CMPLRROOT}/install/include:$CPATH
#
export LD_LIBRARY_PATH=${CMPLRROOT}/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=${CMPLRROOT}/lib:$LIBRARY_PATH
#MIOpen logging
export MIOPEN_ENABLE_LOGGING=1
export MIOPEN_ENABLE_LOGGING_CMD=1
export MIOPEN_LOG_LEVEL=6
#ROCBlas logging
export ROCBLAS_LAYER=2
export ROCBLAS_LOG_PROFILE_PATH=1