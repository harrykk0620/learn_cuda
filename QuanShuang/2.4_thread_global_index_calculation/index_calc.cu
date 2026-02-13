#include <stdio.h>

__global__ void thread_idx_1d_1d()
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    printf("threadIdx.x = %d, blockIdx.x = %d, idx = %d\n", threadIdx.x, blockIdx.x, idx);
}

__global__ void thread_idx_2d_2d()
{
    // 先算 (b_id, t_id)
    int bid = blockIdx.x + blockIdx.y * gridDim.x;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    // idx = bid * 一个block的size + t_id
    int idx = bid * blockDim.x * blockDim.y + tid;

    printf("bid = %d, tid = %d, idx = %d\n", bid, tid, idx);
}

__global__ void thread_idx_2d_3d()
{
    // grid是2d
    int bid = blockIdx.x + blockIdx.y * gridDim.x;
    // block是3d
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int idx = bid * blockDim.x * blockDim.y * blockDim.z + tid;
    printf("bid = %d, tid = %d, idx = %d\n", bid, tid, idx);
}

int main()
{   
    // 都是一维的
    printf("1d grid 1d block\n");
    dim3 block_size(3);
    dim3 grid_size(4); 
    thread_idx_1d_1d<<<grid_size, block_size>>>();
    cudaDeviceSynchronize();
    printf("\n");

    // 2d grid 2d block
    printf("2d grid 2d block\n");
    block_size = dim3(2, 3);
    grid_size = dim3(2, 3);
    thread_idx_2d_2d<<<grid_size, block_size>>>();
    cudaDeviceSynchronize();
    printf("\n");

    // 2d grid 3d block
    printf("2d grid 3d block\n");
    block_size = dim3(2, 3, 4);
    grid_size = dim3(2, 3);
    thread_idx_2d_3d<<<grid_size, block_size>>>();
    cudaDeviceSynchronize();

}