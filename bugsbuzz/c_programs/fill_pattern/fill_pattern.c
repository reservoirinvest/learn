#include<stdio.h>

void fill(int arr[][20], int n);

int main(void) {
// Initialize
    int arr[][20] = {{0}};
    int n = 5, i, j;
    // int n = 20;
    fill(arr, n);
    for (i=0; i <5; i++){
        for(j=0; j<5; j++){
            printf("\t%d", arr[i][j]);
        }
        printf("\n");
    }
    return 0;
}

void fill (int arr[][20], int n){

    int r, c;

    if(n==1){
    arr[0][0] = 1;
    return;
    }

    for (r=0; r<n; r++) {
        arr[r][n-1] = n;
    }

    for(c=0; c<n; c++){
        arr[n-1][c] = n;      
    }

    return fill(arr, n-1);
}