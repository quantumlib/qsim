// Copyright 2023 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SIMULATOR_CUDA2HIP_H_
#define SIMULATOR_CUDA2HIP_H_

#define cublasCaxpy              hipblasCaxpy
#define cublasCdotc              hipblasCdotc
#define cublasCreate             hipblasCreate
#define cublasCscal              hipblasCscal
#define cublasCsscal             hipblasCsscal
#define cublasDestroy            hipblasDestroy
#define cublasDznrm2             hipblasDznrm2
#define cublasHandle_t           hipblasHandle_t
#define cublasScnrm2             hipblasScnrm2
#define CUBLAS_STATUS_SUCCESS    HIPBLAS_STATUS_SUCCESS
#define cublasStatus_t           hipblasStatus_t
#define cublasZaxpy              hipblasZaxpy
#define cublasZdotc              hipblasZdotc
#define cublasZdscal             hipblasZdscal
#define cublasZscal              hipblasZscal
#define cuCimagf                 hipCimagf
#define cuCimag                  hipCimag
#define cuComplex                hipComplex
#define cuCrealf                 hipCrealf
#define cuCreal                  hipCreal
#define CUDA_C_32F               HIPBLAS_C_32F
#define CUDA_C_64F               HIPBLAS_C_64F
#define cudaDeviceSynchronize    hipDeviceSynchronize
#define cudaError_t              hipError_t
#define cudaFree                 hipFree
#define cudaGetErrorString       hipGetErrorString
#define cudaMalloc               hipMalloc
#define cudaMemcpyAsync          hipMemcpyAsync
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost   hipMemcpyDeviceToHost
#define cudaMemcpy               hipMemcpy
#define cudaMemcpyHostToDevice   hipMemcpyHostToDevice
#define cudaMemset               hipMemset
#define cudaPeekAtLastError      hipPeekAtLastError
#define cudaSuccess              hipSuccess
#define cuDoubleComplex          hipDoubleComplex

template <typename T>
__device__ __forceinline__ T __shfl_down_sync(
    unsigned mask, T var, unsigned int delta, int width = warpSize) {
  return __shfl_down(var, delta, width);
}

#endif  // SIMULATOR_CUDA2HIP_H_
