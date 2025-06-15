#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void sum_array_kernel(float* array, float* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(result, array[i]);
    }
}

int main() {
    int n, num_runs;
    
    // Ввод размера массива
    printf("Enter the number of elements in the array (more than 100000): ");
    if (scanf("%d", &n) != 1 || n <= 100000) {
        fprintf(stderr, "Invalid input! Array size should be more than 100000.\n");
        return 1;
    }
    
    // Ввод количества запусков
    printf("Enter the number of runs: ");
    if (scanf("%d", &num_runs) != 1 || num_runs <= 0) {
        fprintf(stderr, "Invalid input for number of runs!\n");
        return 1;
    }

    float* array = (float*)malloc(n * sizeof(float));
    float sum = 0.0f;

    // Инициализация массива случайными значениями
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
    double* times = (double*)malloc(num_runs * sizeof(double));
    int threads_list[] = { 4, 8, 16 }; // Разные варианты количества потоков

    for (int t = 0; t < sizeof(threads_list) / sizeof(threads_list[0]); t++) {
        int threads = threads_list[t];
        int blocks = (n + threads - 1) / threads;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        total_time = 0;
        for (int r = 0; r < num_runs; r++) {
            cudaMemset(d_sum, 0, sizeof(float)); // Сброс суммы перед каждым запуском

            cudaEventRecord(start);
            sum_array_kernel<<<blocks, threads>>>(d_array, d_sum, n);
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
        printf("Total sum: %f\n", sum);

        // Выводим первые и последние 5 запусков (или меньше, если runs < 10)
        int print_count = (num_runs < 5) ? num_runs : 5;
        
        printf("\nFirst %d runs:\n", print_count);
        for (int i = 0; i < print_count; i++) {
            printf("Run %d: %.6f sec\n", i + 1, times[i]);
        }

        if (num_runs > 5) {
            printf("\nLast %d runs:\n", print_count);
            int start_idx = (num_runs - print_count > 0) ? num_runs - print_count : 0;
            for (int i = start_idx; i < num_runs; i++) {
                printf("Run %d: %.6f sec\n", i + 1, times[i]);
            }
        }

        printf("\nAverage time: %.6f sec\n", total_time / num_runs);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Освобождение памяти
    free(times);
    cudaFree(d_array);
    cudaFree(d_sum);
    free(array);

    return 0;
}
