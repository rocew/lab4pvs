#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Функция для заполнения массива единицами
void fill_array_with_ones(int* array, size_t size) {
    for (size_t i = 0; i < size; i++) {
        array[i] = 1;
    }
}

// Функция для последовательного вычисления суммы
long long calculate_sum(const int* array, size_t size) {
    long long sum = 0;
    for (size_t i = 0; i < size; i++) {
        sum += array[i];
    }
    return sum;
}

int main() {
    int array_size, num_runs;
    
    // Ввод размера массива
    printf("Enter the number of elements in the array: ");
    if (scanf("%d", &array_size) != 1 || array_size <= 0) {
        fprintf(stderr, "Invalid input for array size!\n");
        return 1;
    }
    
    // Ввод количества запусков
    printf("Enter the number of runs: ");
    if (scanf("%d", &num_runs) != 1 || num_runs <= 0) {
        fprintf(stderr, "Invalid input for number of runs!\n");
        return 1;
    }
    
    // Выделение памяти для массива
    int* array = (int*)malloc(array_size * sizeof(int));
    if (array == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }

    // Заполнение массива единицами
    fill_array_with_ones(array, array_size);

    double total_time = 0.0;

    // Основной цикл измерений
    for (int run = 0; run < num_runs; run++) {
        clock_t start_time = clock();

        long long sum = calculate_sum(array, array_size);

        clock_t end_time = clock();
        double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;

        total_time += time_taken;
    }

    // Вывод результатов
    double avg_time = total_time / num_runs;

    printf("\nResults:\n");
    printf("Array size: %d\n", array_size);
    printf("Number of runs: %d\n", num_runs);
    printf("Average time per run: %.6f seconds\n", avg_time);

    // Освобождение памяти
    free(array);
    return 0;
}
