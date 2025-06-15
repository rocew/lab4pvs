#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define SIZE (1 << 23)
#define RUNS 100

// CUDA ядро для одного шага Bitonic Sort
__global__ void bitonic_sort_step(int* data, int j, int k, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    unsigned int ixj = i ^ j;

    if (ixj > i) {
        if ((i & k) == 0) {
            if (data[i] > data[ixj]) {
                int temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        }
        else {
            if (data[i] < data[ixj]) {
                int temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        }
    }
}

void generate_random_array(int* arr, int size) {
    for (int i = 0; i < size; i++)
        arr[i] = rand() % 1000000;
}

int main() {
    int* h_array = (int*)malloc(SIZE * sizeof(int));
    int* d_array;

    int threads_list[] = { 4, 8, 16 };

    cudaMalloc((void**)&d_array, SIZE * sizeof(int));

    for (int t = 0; t < sizeof(threads_list) / sizeof(threads_list[0]); t++) {
        int threads = threads_list[t];
        int blocks = (SIZE + threads - 1) / threads;

        float times[RUNS];
        float total_time = 0.0f;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for (int r = 0; r < RUNS; r++) {
            generate_random_array(h_array, SIZE);
            cudaMemcpy(d_array, h_array, SIZE * sizeof(int), cudaMemcpyHostToDevice);

            cudaEventRecord(start);

            for (int k = 2; k <= SIZE; k <<= 1) {
                for (int j = k >> 1; j > 0; j >>= 1) {
                    bitonic_sort_step << <blocks, threads >> > (d_array, j, k, SIZE);
                    cudaDeviceSynchronize();
                }
            }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);
            times[r] = ms / 1000.0f;  // перевод в секунды
            total_time += times[r];
        }

        printf("\nThreads: %d\n", threads);
        printf("Array size: %d\n", SIZE);

        printf("First 5 runs:\n");
        for (int i = 0; i < 5; i++) {
            printf("Run %d: %.6f sec\n", i + 1, times[i]);
        }

        printf("\nLast 5 runs:\n");
        for (int i = RUNS - 5; i < RUNS; i++) {
            printf("Run %d: %.6f sec\n", i + 1, times[i]);
        }

        printf("\nAverage time: %.6f sec\n", total_time / RUNS);

        cudaDeviceSynchronize();
    }

    cudaFree(d_array);
    free(h_array);
    return 0;
}
