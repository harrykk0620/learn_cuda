#include <stdio.h>

__global__ void HelloWorld()
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int idx = tid + bid * blockDim.x;
    printf("Hello world from GPU. Block %d, Thread %d, index %d)\n", bid, tid, idx);
}

int main()
{
    printf("Hello world from CPU\n");
    HelloWorld<<<2, 2>>>();
    cudaDeviceSynchronize();
}