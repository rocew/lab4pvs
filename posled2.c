#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

int safe_input(const char* prompt) {
    int value;
    while (1) {
        printf("%s", prompt);
        if (scanf("%d", &value) == 1 && value > 0) {
            break;
        }
        printf("Invalid input. Please enter a positive integer.\n");
        // Clear input buffer
        while (getchar() != '\n');
    }
    return value;
}

double get_time_ms() {
    static LARGE_INTEGER freq;
    static BOOL initialized = FALSE;
    LARGE_INTEGER now;
    if (!initialized) {
        QueryPerformanceFrequency(&freq);
        initialized = TRUE;
    }
    QueryPerformanceCounter(&now);
    return (1000.0 * now.QuadPart) / freq.QuadPart;
}

void merge(int* arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    int* L = malloc(n1 * sizeof(int));
    int* R = malloc(n2 * sizeof(int));

    for (int i = 0; i < n1; ++i) L[i] = arr[l + i];
    for (int j = 0; j < n2; ++j) R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;

    while (i < n1 && j < n2)
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];

    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];

    free(L);
    free(R);
}

void mergeSort(int* arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

void generate_random_array(int* arr, int size) {
    for (int i = 0; i < size; ++i)
        arr[i] = rand() % 1000000;
}

int main() {
    // Get user input
    int SIZE = safe_input("Enter array size: ");
    int RUNS = safe_input("Enter number of runs: ");

    int* arr = malloc(SIZE * sizeof(int));
    if (arr == NULL) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    double* times = malloc(RUNS * sizeof(double));
    if (times == NULL) {
        printf("Memory allocation failed!\n");
        free(arr);
        return 1;
    }

    printf("\nRunning tests for %d elements, %d iterations...\n", SIZE, RUNS);

    for (int i = 0; i < RUNS; ++i) {
        generate_random_array(arr, SIZE);
        double start = get_time_ms();

        mergeSort(arr, 0, SIZE - 1);

        double end = get_time_ms();
        times[i] = (end - start) / 1000.0; // Store time in seconds

        // Print progress every 10%
        if ((i + 1) % (RUNS / 10) == 0 || i == RUNS - 1) {
            printf("Completed: %d/%d (%.0f%%)\n", i + 1, RUNS, (i + 1) * 100.0 / RUNS);
        }
    }

    printf("\nResults (first and last 5 runs):\n");
    int show_runs = (RUNS < 10) ? RUNS : 5;
    
    printf("First %d runs:\n", show_runs);
    for (int i = 0; i < show_runs; ++i) {
        printf("Run %d: %.6f sec\n", i + 1, times[i]);
    }

    if (RUNS > 10) {
        printf("\nLast %d runs:\n", show_runs);
        for (int i = RUNS - show_runs; i < RUNS; ++i) {
            printf("Run %d: %.6f sec\n", i + 1, times[i]);
        }
    }

    double total = 0;
    for (int i = 0; i < RUNS; ++i)
        total += times[i];

    printf("\nAverage execution time (Merge Sort): %.6f sec\n", total / RUNS);

    // Cleanup
    free(arr);
    free(times);
    return 0;
}
