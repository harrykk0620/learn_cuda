#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// CUDA Kernel 函数：在 GPU 设备端执行的函数
// __global__ 是 CUDA 的一个特殊函数限定符，表示这是一个 kernel 函数
// 可以被主机端代码启动，并在 GPU 的多个线程上并行执行
__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{
    // 获取当前线程的全局索引 (global thread index)
    // blockIdx.x 是当前线程块的索引
    // blockDim.x 是每个线程块中的线程数
    // threadIdx.x 是当前线程在其所属线程块内的索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 检查索引是否超出了数组边界，防止越界访问
    // 这很重要，因为线程总数可能比数组长度 N 略大
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// 主机端 Host 函数
int main(int argc, char **argv)
{
    const int N = 1 << 25; // 33,554,432 个元素，与 CPU 版本一致
    printf("Size of array is %d\n", N);

    // 1. 在 Host (CPU) 端分配内存，并初始化数据
    size_t bytes = N * sizeof(float); // 计算需要的字节数，方便后续使用
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);

    // --- 初始化 Host 数组 A 和 B ---
    for (int i = 0; i < N; i++)
    {
        h_a[i] = rand() / (float)RAND_MAX; // 随机生成 [0, 1] 区间内的浮点数
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // 2. 在 Device (GPU) 端分配内存
    float *d_a, *d_b, *d_c; // d_ 通常是 "device" 的缩写
    cudaMalloc((void**)&d_a, bytes); // 在 GPU 显存上为 d_a 分配 bytes 字节的空间
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_c, bytes);

    // 3. 将 Host 数据拷贝到 Device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice); // 将 h_a 的数据复制到 d_a
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice); // 将 h_b 的数据复制到 d_b

    // 4. 配置 Kernel 启动参数
    int blockSize = 256; // 每个线程块包含 256 个线程 (常见选择，性能较好)
    int gridSize = (N + blockSize - 1) / blockSize; // 计算需要多少个线程块
    // (N + blockSize - 1) / blockSize 是向上取整的技巧，确保至少有 N 个线程

    printf("Grid size: %d blocks, Block size: %d threads\n", gridSize, blockSize);
    printf("Thread数量有%d个\n", blockSize * gridSize);

    // 5. 启动 Kernel 函数
    // 创建CUDA事件用于精确计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 记录开始时间
    cudaEventRecord(start);
    // 这行代码告诉 GPU 在 gridSize 个块，每个块有 blockSize 个线程的配置下运行 sumArraysOnGPU
    sumArraysOnGPU<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    // 记录结束时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // 等待GPU完成所有工作
    
    // 计算经过的时间
    float elapsed_time_gpu_ms;
    cudaEventElapsedTime(&elapsed_time_gpu_ms, start, stop);

    printf("GPU calculation time: %f ms\n", elapsed_time_gpu_ms);
    
    // 销毁事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 6. 将 Device 结果拷贝回 Host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost); // 将 d_c 的结果复制回 h_c

    // 7. (可选) 验证结果 (检查几个点)
    bool correct = true;
    for (int i = 0; i < 10; ++i) { // 检查前10个元素
        if (fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5) { // 使用小的误差容忍度
            correct = false;
            break;
        }
    }
    if (!correct) {
        printf("Result verification FAILED.\n");
    } else {
        printf("Result verification PASSED.\n");
    }

    // 8. 释放 Device 端内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // 9. 释放 Host 端内存
    free(h_a);
    free(h_b);
    free(h_c);

    return EXIT_SUCCESS;
}