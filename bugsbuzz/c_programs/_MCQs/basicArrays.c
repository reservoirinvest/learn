#include <stdio.h>
#include <string.h>

int main (void){
    int arr[10][10] = {{0}};
    int i, j;
    for(i=0; i<10; i++){
        for(j=0; j<10; j++){
            printf("\t%d ", arr[i][j]);
            // printf("\t%d ", arr[0][i]);
        }
        printf("\n");
    }

    return 0;
}