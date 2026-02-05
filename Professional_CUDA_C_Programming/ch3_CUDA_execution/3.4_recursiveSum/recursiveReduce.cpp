#include <iostream>
#include <ctime>
using namespace std;

// 实现recursiveReduce求和，cpp算法不要求长度是2^N，2^N计算效率会比较高
void initialData(int *data, const int size)
{
    for (int i = 0; i < size; i++)
    {
        data[i] = 1;
    }
}

int recursiveReduceStride(int *data, const int size)
{ 
    if (size == 1)
        return data[0];
    if (size % 2 == 1)
    {
        data[0] += data[size - 1]; // 奇数的话，将最后一个数加到第一个数
    }
    int stride = size / 2; // 跨度
    for (int i = 0; i < stride; i++)
    {
        data[i] += data[i + stride]; // 偶数个数，将中间数加到两边数
    }
    return recursiveReduceStride(data, stride);   
}

int main(int argc, char **argv)
{
    int size = (1 << 24) + 1;
    int *data =  new int[size];
    initialData(data, size);
    int *temp = new int[size];

    // 交错配对
    copy(data, data + size, temp);
    clock_t start, end;
    start = clock();
    int stride_result = recursiveReduceStride(temp, size);
    end = clock();
    cout << "recursiveReduceOnHost time: " << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;
    cout << "recursiveReduceOnHost: " << stride_result << endl;
    if (size != stride_result)
    {
        cout << "recursiveReduceOnHost: result false" << endl;
    }
    else
    {
        cout << "recursiveReduceOnHost: result true" << endl;
    }


    // 释放内存
    if (data != NULL)
    {
        delete[] data;
        data = NULL;
    }
    if (temp != NULL)
    {
        delete[] temp;
        temp = NULL;
    }
}

