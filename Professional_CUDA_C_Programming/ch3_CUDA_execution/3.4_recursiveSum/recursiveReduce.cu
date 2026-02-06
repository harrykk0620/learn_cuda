#include "../../common/common.h"
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 1024

const int MAX = 10;
const int MIN = 0;
void initialData(int *data, const int size)
{
    for (int i = 0; i < size; i++)
    {
        data[i] = MIN + rand() % (MAX - MIN + 1);
        // data[i] = 1;
    }
}

long long recursiveReduce(long long  *data, const int size)
{
    if (size == 1)
        return data[0];
    int stride = size / 2; 
    for (int i = 0; i < stride; i++)
    {
        data[i] += data[i + stride];
    }
    return recursiveReduce(data, stride);
}

// 设计算法时是按照Block设计的，但核函数是按照thread写的
// 要绘制一个清晰的block算法图
__global__ void recursiveReduceOnGPU(int *g_idata, int *g_odata, const int size)
{
    // 处理单个块的和
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    
    // 边界检查
    int idx = block_id * blockDim.x + thread_id;
    if (idx >= size) return;
    
    // 处理块内数据
    int* block_array = g_idata + blockIdx.x * blockDim.x; // block array的指针
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) // stride=0说明block求和完成
    {
        // 错位求和，每次都把求和结果放在最前面，减少了warp divergence(1024线程为例，前16个warp进if分支求和，后16个不进)
        if (thread_id < stride)
        {
            // 这里每次都从global memory中读取数据，不高效
            block_array[thread_id] += block_array[thread_id + stride]; // 每个block一定要被填满
        }
        __syncthreads(); // 确保了 Block 内的所有线程在继续执行下一条指令之前，都到达了这个同步点
    }

    // 往g_odata输出
    g_odata[block_id] = block_array[0];
}

// 使用共享内存
__global__ void recursiveReduceOnGPUShare(int *g_idata, int *g_odata, const int size)
{
    // 处理单个块的和
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    
    // 边界检查
    int idx = block_id * blockDim.x + thread_id;
    if (idx >= size) return;
    
    // 共享内存加载数据
    __shared__ int s_data[BLOCK_SIZE]; // 报错，CUDA只支持这种静态大小的共享内存声明
    int* block_array = g_idata + blockIdx.x * blockDim.x; // block array的指针
    if (idx < size)
    {
        s_data[thread_id] = block_array[thread_id];
    }
    else
    {
        s_data[thread_id] = 0;
    }
    __syncthreads(); // 确保所有数据加载到共享内存

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) // stride=0说明block求和完成
    {
        if (thread_id < stride)
        {
            // 这里每次都从global memory中读取数据，不高效
            s_data[thread_id] += s_data[thread_id + stride]; // 每个block一定要被填满
        }
        __syncthreads(); // 确保了 Block 内的所有线程在继续执行下一条指令之前，都到达了这个同步点
    }

    // 往g_odata输出
    g_odata[block_id] = s_data[0];
}

// 使用共享内存，使用add during loading，第一次loading时就完成一次reduce add
__global__ void recursiveReduceOnGPUShare1(int *g_idata, int *g_odata, const int size)
{
    // 处理单个块的和
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    
    // 边界检查
    int idx = block_id * blockDim.x + thread_id;
    if (idx >= size) return;
    
    // 共享内存加载数据
    __shared__ int s_data[BLOCK_SIZE/2];
    int* block_array = g_idata + blockIdx.x * blockDim.x; // block array的指针
    if (thread_id < BLOCK_SIZE / 2)
    {
        if (idx < size)
        {
            s_data[thread_id] = block_array[thread_id];
            if (idx + BLOCK_SIZE / 2 < size)
            {
                s_data[thread_id] +=  block_array[thread_id + BLOCK_SIZE / 2];
            }
        }
        else
        {
            s_data[thread_id] = 0;
        }   
    }

    __syncthreads(); // 确保所有数据加载到共享内存

    for (int stride = BLOCK_SIZE / 4; stride > 0; stride /= 2) // stride=0说明block求和完成
    {
        if (thread_id < stride)
        {
            // 这里每次都从global memory中读取数据，不高效
            s_data[thread_id] += s_data[thread_id + stride]; // 每个block一定要被填满
        }
        __syncthreads(); // 确保了 Block 内的所有线程在继续执行下一条指令之前，都到达了这个同步点
    }

    // 往g_odata输出
    g_odata[block_id] = s_data[0];
}

int main(int argc, char **argv) // argc是命令行参数数量，至少为1，argv[0]是程序本身名称
{
    int size = 1 << 26;
    printf("Vector size %d\n", size);
    
    // CPU
    double start, end;
    // 分配内存
    start = seconds();
    int* data = (int*)malloc(size * sizeof(int));
    initialData(data, size);
    long long* temp = (long long *)malloc(size * sizeof(long long));
    for (int i = 0; i < size; i++)
    {
        temp[i] = (long long)data[i];
    }
    long long cpu_sum = recursiveReduce(temp, size);
    end = seconds();
    printf("CPU sum \t: %lld, time: %f (ms)\n", cpu_sum, (end - start) * 1000);

    // GPU
    /*
    理解cudaError_t cudaMalloc ( void** devPtr, size_t size )
    void* 是一个无类型指针，可以指向任何一个数据类型。
    void** (两个星号):
        这是一个指向 void* 的指针。它是指针的指针。因为 cudaMalloc 函数需要修改一个 void* 类型的变量（也就是那个用来保存设备地址的变量）。
    */ 

    // int block_size = 32;
    // if (argc > 1)
    // {
    //     block_size = atoi(argv[1]);
    // }
    // if (size % block_size != 0)
    // {
    //     printf("size %d must be a multiple of block_size %d\n", size, block_size);
    //     exit(1);
    // }

    int block_size = BLOCK_SIZE; // shared memory只能接收静态数据大小，不能动态分配

    // 普通GPU错位求和
    start = seconds();
    int *d_temp;
    cudaMalloc(&d_temp, size * sizeof(int));
    cudaMemcpy(d_temp, data, size * sizeof(int), cudaMemcpyHostToDevice);
    // 思路：每个block求一个和到d_odata中，d_odata复制到h_block_sum中，cpu循环求和
    
    int *d_odata;

    int grid_size = size / block_size;
    cudaMalloc(&d_odata, grid_size * sizeof(int));
    recursiveReduceOnGPU<<<grid_size, block_size>>>(d_temp, d_odata, size);
    cudaDeviceSynchronize();
    int* h_block_sum = (int*)malloc(grid_size * sizeof(int));
    cudaMemcpy(h_block_sum, d_odata, grid_size * sizeof(int), cudaMemcpyDeviceToHost);
    long long gpu_sum = 0;
    for (int i = 0; i < grid_size; i++)
    {
        gpu_sum += h_block_sum[i];
    }
    end = seconds();
    printf("GPU sum0\t: %lld, time: %f (ms)\n", gpu_sum, (end - start) * 1000);
    cudaFree(d_temp);
    cudaFree(d_odata);
    free(h_block_sum);

    // 使用共享内存
    start = seconds();
    cudaMalloc(&d_temp, size * sizeof(int));
    cudaMemcpy(d_temp, data, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_odata, grid_size * sizeof(int));
    recursiveReduceOnGPUShare<<<grid_size, block_size>>>(d_temp, d_odata, size);
    cudaDeviceSynchronize();
    h_block_sum = (int*)malloc(grid_size * sizeof(int));
    cudaMemcpy(h_block_sum, d_odata, grid_size * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid_size; i++)
    {
        gpu_sum += h_block_sum[i];
    }
    end = seconds();
    printf("GPU sum1\t: %lld, time: %f (ms)\n", gpu_sum, (end - start) * 1000);

    // 使用共享内存，使用add during loading
    start = seconds();
    cudaMalloc(&d_temp, size * sizeof(int));
    cudaMemcpy(d_temp, data, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_odata, grid_size * sizeof(int));
    recursiveReduceOnGPUShare1<<<grid_size, block_size>>>(d_temp, d_odata, size);
    cudaDeviceSynchronize();
    h_block_sum = (int*)malloc(grid_size * sizeof(int));
    cudaMemcpy(h_block_sum, d_odata, grid_size * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid_size; i++)
    {
        gpu_sum += h_block_sum[i];
    }
    end = seconds();
    printf("GPU sum2\t: %lld, time: %f (ms)\n", gpu_sum, (end - start) * 1000);

    free(data);
    free(temp);
    free(h_block_sum);
    cudaFree(d_temp);
    cudaFree(d_odata);
}
