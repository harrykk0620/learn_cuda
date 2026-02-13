#include "../common.h"
#include <stdio.h>

__global__ void badKernel(int *a) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  *a = idx;
}

int main() 
{
    int *a = NULL;
    printf("address of a after init is %p\n", a);
    cudaMalloc(&a, sizeof(int)); // cudaMalloc第一个是要双重指针。cudaMalloc就是在函数内部修改你传入的指针变量，使其指向新分配的 GPU 设备内存
    printf("address of a after malloc is %p\n", a);
    badKernel<<<1, 1>>>(NULL);  // 直接调用核函数。注意launch kernal没有返回值，不能CUDA_CHECK
    
    
    printf("CUDA_CHECK(cudaGetLastError())\n");
    CUDA_CHECK(cudaGetLastError());  // 检查核函数启动错误
    /*
    cudaGetLastError() 只返回最近一次 CUDA Runtime API 调用的错误。最近一次调用是 badKernel<<<>>>，它成功了
    */
    cudaDeviceSynchronize(); // GPU 的错误状态不会自动“推送”给 CPU，必须由 CPU 主动发起一个 同步操作（synchronization operation）去“拉取”错误。
    printf("CUDA_CHECK(cudaGetLastError()) after cudaDeviceSynchronize\n");
    CUDA_CHECK(cudaGetLastError());
    printf("CUDA_CHECK(cudaDeviceSynchronize())\n");
    CUDA_CHECK(cudaDeviceSynchronize());  // 检查设备同步错误
    printf("CUDA_CHECK(cudaGetLastError())\n");
    CUDA_CHECK(cudaGetLastError());  // 检查核函数启动错误
}