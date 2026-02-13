// cuda核函数编写流程
// 注意：核函数不支持C++的iostream

// NVIDIA 的 CUDA 文档明确指出，要在 device code 中使用 printf，必须包含 stdio.h
#include <stdio.h>

__global__ void hello_from_gpu()
{
    printf("Hello World from GPU!\n");
}

/*
int main()
{
    主机代码
    核函数调用
    主机代码
    return 0;
}
*/

int main()
{
    /*
    <<<1,1>>>: 这是 CUDA 特有的执行配置语法（execution configuration syntax），它告诉 GPU 如何启动这个核函数。它包含三个参数，用逗号分隔，放在三个尖括号里。这里只用了前两个参数：
    第一个 1: 这是 gridDim.x，表示网格（Grid）中块（Block）的数量。这里设置为 1，意味着整个计算任务被组织成 1 个网格，网格里有 1 个块。
    第二个 1: 这是 blockDim.x，表示块（Block）中线程（Thread）的数量。这里设置为 1，意味着这个唯一的块里只有 1 个线程。
    第三个参数（未指定）: 是 sharedMem，用于指定每个块动态分配的共享内存（Shared Memory）大小，默认为 0。
    第四个参数（未指定）: 是 streamId，用于指定 CUDA 流（Stream ID），默认为 0（即默认流）
    */
    hello_from_gpu<<<4,4>>>();

    // 让CPU线程等待核函数执行完成
    cudaDeviceSynchronize();
    return 0;
}