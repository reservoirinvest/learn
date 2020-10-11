#include<stdio.h>
#include<stdlib.h>

void init(int arr[100][100]);

int main(){

    int flight[100][100];
    int i, j;

    init(flight);

     for (i = 0; i< 100; i++){
        for (j = 0; j< 100; j++)
            printf("init(flight[%d,%d] array is %d", i, j, init(flight[i][j]));
     }
    return 0;
}

void init(int arr[100][100]) {

    int i, k;

    for (int i = 0; i <100; i++){
        for(int k = 0; k < 100; k++){
            arr[i][k] = rand();
        }    
    }
   
}