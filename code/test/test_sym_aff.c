#include <tsne.h>
#include <stdio.h>

static void testGetSquaredEuclideanDistances(void) {
    float x[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    int n = 3;
    int d = 3;
    float res[9];

    getSquaredEuclideanDistances((float*)x, n, d, (float*)res);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i * n + j;
            printf("%.2lf ", res[idx]);
        }
        printf("\n");
    }
}

int main() {
    testGetSquaredEuclideanDistances();
}