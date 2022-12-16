#include<stdio.h>

int smartAlecArray(void);
int thislooksgoodbutdoesnotwork (void);

int main(void){
    int arr[2][3], i;
    for(i=0; i<2; i++){
        arr[i] = {1, 1, 1}; //this is pure EVIL!!!. Doesn't even compile
    }
    printf("Let me pause and take a fresh breath\n");
    return 0;
}


int thislooksgoodbutdoesnotwork (void) {
    int arr[2][3], arr2[3] = {1, 1, 1}, i;
    for (i=0; i<2; i++)
        arr[i] = arr2; //this bugger is EVIL!!! Does'nt compile

    printf("Let me pause here for the dodo");

}

int smartAlecArray (void){
    int arr[2][3] = {{0}};
    int i, j;
    while(i<2){
        j = 0;
        while (j<3){
            arr[i][j]++;
            j++;
        }
        i++;
    }
    printf("pause here to check the array!");
}