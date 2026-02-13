// 推广到多线程
/*
1. cuda可以组织三维的网格和线程块
2. blockIdx和threadIdx都是一个结构体（类型为unit3），有x,y,z三个成员组成。blockIdx.x, blockIdx.y, blockIdx.z; threadIdx.x, threadIdx.y, threadIdx.z
3. gridDim和blockDim也都是结构体（类型为dim3），有xyz三个成员
4. 取值范围
    blockIdx.x: 0 ~ gridDim.x - 1
    blockIdx.y: 0 ~ gridDim.y - 1
    blockIdx.z: 0 ~ gridDim.z - 1

    threadIdx.x: 0 ~ blockDim.x - 1
    threadIdx.y: 0 ~ blockDim.y - 1
    threadIdx.z: 0 ~ blockDim.z - 1

Dim和Idx都是内建变量，只在核函数有效，无需定义

网格大小限制：
    gridDim.x <= 2147483648, gridDim.y <= 65535, gridDim.z <= 65535
线程块大小限制：
    blockDim.x <= 1024, blockDim.y <= 1024, blockDim.z <= 64
    线程块总的大小不能超过1024
*/

#include <stdio.h>

__global__ void kernel_func()
{
    if (blockIdx.x + blockIdx.y + blockIdx.z + threadIdx.x + threadIdx.y + threadIdx.z == 0)
    {
        printf("输出grid的size\n");
        printf("gridDim.x = %d, gridDim.y = %d, gridDim.z = %d\n", gridDim.x, gridDim.y, gridDim.z);
        printf("blockDim.x = %d, blockDim.y = %d, blockDim.z = %d\n", blockDim.x, blockDim.y, blockDim.z);
    }

    printf("blockIdx.x = %d, blockIdx.y = %d, blockIdx.z = %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
    printf("threadIdx.x = %d, threadIdx.y = %d, threadIdx.z = %d\n", threadIdx.x, threadIdx.y, threadIdx.z);
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int bid = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    printf("(bid, tid) is (%d, %d)\n", bid, tid);
}

int main()
{
    dim3 grid_size(1,2,3);
    dim3 block_size(1,2,3);
    kernel_func<<<grid_size, block_size>>>();
    cudaDeviceSynchronize();
}