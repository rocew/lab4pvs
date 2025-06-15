#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define runs 100
__global__ void sum_array_kernel(float* array, float* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(result, array[i]);
    }
}

int main() {
    int n = 10000000; // Более 100000 элементов
    float* array = (float*)malloc(n * sizeof(float));
    float sum = 0.0f;

    // Инициализация массива
    for (int i = 0; i < n; i++) {
        array[i] = rand() / (float)RAND_MAX;
    }

    // Выделение памяти на GPU
    float* d_array;
    float* d_sum;
    cudaMalloc((void**)&d_array, n * sizeof(float));
    cudaMalloc((void**)&d_sum, sizeof(float));

    // Копирование данных на GPU
    cudaMemcpy(d_array, array, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_sum, 0, sizeof(float)); // Инициализация суммы на GPU

    double total_time = 0;
    double times[runs];
    int threads_list[] = { 4,8,16 }; // Разные варианты количества потоков

    for (int t = 0; t < sizeof(threads_list) / sizeof(threads_list[0]); t++) {
        int threads = threads_list[t];
        int blocks = (n + threads - 1) / threads;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        total_time = 0;
        for (int r = 0; r < runs; r++) {
            cudaMemset(d_sum, 0, sizeof(float)); // Сброс суммы перед каждым запуском

            cudaEventRecord(start);
            sum_array_kernel << <blocks, threads >> > (d_array, d_sum, n);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);
            times[r] = ms / 1000.0; // Переводим в секунды
            total_time += times[r];
        }

        // Копируем результат обратно на CPU для проверки
        cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

        printf("\nThreads: %d\n", threads);
        printf("Array size: %d\n", n);

        printf("First 5 runs:\n");
        for (int i = 0; i < 5; i++) {
            printf("Run %d: %.6f sec\n", i + 1, times[i]);
        }

        printf("\nLast 5 runs:\n");
        for (int i = runs - 5; i < runs; i++) {
            printf("Run %d: %.6f sec\n", i + 1, times[i]);
        }

        printf("\nAverage time: %.6f sec\n", total_time / runs);
    }

    // Освобождение памяти
    cudaFree(d_array);
    cudaFree(d_sum);
    free(array);

    return 0;
}