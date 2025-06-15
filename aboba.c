#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

#define SIZE 8388608
#define RUNS 100

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
    int* arr = malloc(SIZE * sizeof(int));
    double times[RUNS];

    for (int i = 0; i < RUNS; ++i) {
        generate_random_array(arr, SIZE);
        double start = get_time_ms();

        mergeSort(arr, 0, SIZE - 1);

        double end = get_time_ms();
        times[i] = end - start;

        if (i < 5 || i >= RUNS - 5) {
            printf("Run %d: %.6f sec\n", i + 1, times[i] / 1000.0);
        }
    }

    double total = 0;
    for (int i = 0; i < RUNS; ++i)
        total += times[i];

    printf("Average Time (Merge Sort): %.6f sec\n", total / RUNS / 1000.0);

    free(arr);
    return 0;
}
