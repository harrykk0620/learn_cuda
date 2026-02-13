#include <stdio.h>

__global__ void kernel_func()
{
    printf("gridDim.x = %d, blockDim.x = %d\n", gridDim.x, blockDim.x);

    printf("blockIdx.x = %d, threadIdx.x = %d\n", blockIdx.x, threadIdx.x);

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 唯一表示
    printf("idx = %d\n", idx);
}

int main()
{
    kernel_func<<<2, 4>>>();
    cudaDeviceSynchronize();
}