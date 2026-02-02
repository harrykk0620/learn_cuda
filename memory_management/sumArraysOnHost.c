#include <stdio.h> // 标准输入输出库，用于 printf 等函数
#include <stdlib.h>  // 用于 rand() 和 RAND_MAX
#include <time.h> // 包含 time.h 以使用 clock(), clock_t

// 函数声明：定义了一个在 host (CPU) 上运行的函数，用于执行数组求和
void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    // 使用 for 循环遍历数组 A 和 B 的每一个元素
    for (int idx = 0; idx < N; ++idx)
    {
        // 将 A[idx] 和 B[idx] 相加的结果存储到 C[idx]
        C[idx] = A[idx] + B[idx];
    }
}

// main 函数是程序的入口点
int main()
{
    // 定义常量 N，表示数组的大小（元素个数）
    const int N = 1 << 25; // 1 左移 25 位，即 2^25 = 33,554,432 个元素
                           // 这是一个相当大的数组，有助于观察并行计算的优势
    printf("size of array is %d\n", N);
    /*
    C语言内存分配
    void *malloc(size_t size)
    size_t size: 这是你想要分配的内存大小，以字节 (bytes) 为单位。size_t 是一个无符号整数类型
    void *: 这是函数的返回值。malloc 成功时，会返回一个指向新分配的内存块起始地址的指针。
            这个指针的类型是 void *，它是一个通用指针 (generic pointer)，可以指向任何类型的数据。
    */
    // 在 host (CPU) 内存上分配三个 float 类型数组的空间
    // A, B 是输入数组，C 是输出数组
    // h_a都是内存地址
    float *h_a = (float *)malloc(N * sizeof(float)); // h_ 通常是 "host" 的缩写
    float *h_b = (float *)malloc(N * sizeof(float));
    float *h_c = (float *)malloc(N * sizeof(float));
    printf("h_a是指针：%p\n", h_a);

    // --- 初始化数组 A 和 B ---
    for (int i = 0; i < N; i++)
    {
        h_a[i] = rand() / (float)RAND_MAX; // 随机生成 [0, 1] 区间内的浮点数
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // 调用上面定义的函数，在 host 上执行数组求和运算
    clock_t start_time = clock(); 
    sumArraysOnHost(h_a, h_b, h_c, N);
    clock_t end_time = clock(); 
    double elapsed_time_ms = (double)(end_time - start_time) * 1000.0 / CLOCKS_PER_SEC;
    printf("Host calculation time: %f ms\n", elapsed_time_ms);

    // 释放之前动态分配的 host 内存空间，防止内存泄漏
    free(h_a);
    free(h_b);
    free(h_c);

    return EXIT_SUCCESS; // 程序正常退出，返回 0 (EXIT_SUCCESS 通常定义为 0)
}