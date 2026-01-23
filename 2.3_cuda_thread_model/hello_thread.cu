// cuda核函数编写流程
// 注意：核函数不支持C++的iostream

// NVIDIA 的 CUDA 文档明确指出，要在 device code 中使用 printf，必须包含 stdio.h
#include <stdio.h>
#include <cuda_runtime.h> // Required for CUDA runtime API (including events)

// A simple kernel that does some work (more than just printf)
// Using a loop to make timing more measurable
__global__ void workload_kernel() {
    int result = 0;
    for (int i = 0; i <10000; i++)
    {
        result += i;
    }
}

int main()
{
    cudaSetDevice(1);
    // --- Configuration 1: <<<1024, 32>>> ---
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    cudaEventRecord(start1);
    workload_kernel<<<1024, 32>>>();
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1); // Ensure kernel finishes before measuring

    float milliseconds1 = 0;
    cudaEventElapsedTime(&milliseconds1, start1, stop1);
    printf("Configuration 1 (<<<1024, 32>>>): %f ms\n", milliseconds1);

    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);

    // --- Configuration 2: <<<32, 1024>>> ---
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    cudaEventRecord(start2);
    workload_kernel<<<32, 1024>>>();
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);

    float milliseconds2 = 0;
    cudaEventElapsedTime(&milliseconds2, start2, stop2);
    printf("Configuration 2 (<<<32, 1024>>>): %f ms\n", milliseconds2);

    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    // --- Configuration 3: <<<2, 4>>> (Your original example) ---
    cudaEvent_t start3, stop3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);

    cudaEventRecord(start3);
    workload_kernel<<<2, 4>>>(); // Same workload, fewer threads
    cudaEventRecord(stop3);
    cudaEventSynchronize(stop3);

    float milliseconds3 = 0;
    cudaEventElapsedTime(&milliseconds3, start3, stop3);
    printf("Configuration 3 (<<<2, 4>>>): %f ms\n", milliseconds3);

    cudaEventDestroy(start3);
    cudaEventDestroy(stop3);

    // Final sync to ensure all kernels have finished before program exits
    cudaDeviceSynchronize();

    return 0;
}