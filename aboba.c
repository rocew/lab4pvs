#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ARRAY_SIZE 10000000    // 1 миллион элементов
#define NUM_RUNS 100          // Количество запусков

// Заполнение массива единицами
void fill_array_with_ones(int* array, size_t size) {
    for (size_t i = 0; i < size; i++) {
        array[i] = 1;
    }
}

// Последовательное вычисление суммы
long long calculate_sum(const int* array, size_t size) {
    long long sum = 0;
    for (size_t i = 0; i < size; i++) {
        sum += array[i];
    }
    return sum;
}

int main() {
    int* array = (int*)malloc(ARRAY_SIZE * sizeof(int));
    if (array == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }

    fill_array_with_ones(array, ARRAY_SIZE);

    double total_time = 0.0;

    for (int run = 0; run < NUM_RUNS; run++) {
        clock_t start_time = clock();

        long long sum = calculate_sum(array, ARRAY_SIZE);

        clock_t end_time = clock();
        double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;  // в секундах

        total_time += time_taken;

        // (Опционально) Вывод суммы для проверки (должно быть 1000000)
        // printf("Run %3d: Sum = %lld, Time = %.6f s\n", run + 1, sum, time_taken);
    }

    double avg_time = total_time / NUM_RUNS;

    printf("Array size: %d\n", ARRAY_SIZE);
    printf("Number of runs: %d\n", NUM_RUNS);
    printf("Average time per run: %.6f seconds\n", avg_time);
    printf("Total time for %d runs: %.6f seconds\n", NUM_RUNS, total_time);

    free(array);
    return 0;
}