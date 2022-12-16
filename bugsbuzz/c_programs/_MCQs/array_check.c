# include<stdio.h>

int main(void){
    int arr[5] = {2, 1, 4, 0, 3};

    int i, j;

    for (i=0; i<5; i++){
        for (j = i+1; j < 5; j++) {
            arr[(i+j)%5] = arr[(i+j)%5] + (arr[i] + arr[j])%5;
        }
    }
    
    for(i=0; i<5; i++){
        printf("%d\n", arr[i]);
    }

    return 0;
}