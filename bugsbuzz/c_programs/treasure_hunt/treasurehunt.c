#include <stdio.h>

int sumtreasures(int[5][5], int, int, int, int);

int main(void) {
    int arr[5][5] = {{1, 3, 3, 1, 1},
                     {1, 6, 1, 2, 1},
                     {2, 1, 8, 1, 5},
                     {1, 1, 8, 1, 5},
                     {1, 4, 1, 1, 4}};

    int r1 = 2, r2 = 2, c1 = 4, c2 = 3, sum = 0;

    sum = sumtreasures(arr, r1, r2, c1, c2);

    printf("The size of treasure is: %d\n", sum);

    return 0;
}

// Sum of values of the squares given within the co-ordinates provided.
// It comes to 16
int sumtreasures(int arr[][5], int r1, int r2, int c1, int c2) {
    int i, j, sum = 0;

    for (i = r1; i <= c1; i++) {
        for (j = r2; j <= c2; j++) {
            sum = sum + arr[i][j];
        }
    }
    return sum;
}

// Treasurehunt
