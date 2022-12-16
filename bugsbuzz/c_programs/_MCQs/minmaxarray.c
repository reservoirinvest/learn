#include<stdio.h>
/* Main take-aways:
   ---------------

   1) Even for an int function, I can return two or more values, if I pass the address of the variable to be returned as a pointer.
   2) I can initialize min and max to the same content (value) of arr[0].
   3) *max = arr[0] is a valid declaration of a pointer to an array.  The array can be traversed:
      a) either by using the pointer,
      b) or by using the address of the array.

*/

int minmax(int [], int, int*);

int main(void){
    int min, max = 0;
    int arr[] = {3,2,1,4,5};

    min = minmax(arr, 5, &max);

    printf("%d\n", min);
    printf("%d\n", max);

    return 0;
}

int minmax(int arr[], int size, int *max){
    int i, min;

    min =arr[0];
    *max= arr[0];

    for(i=1; i<size; i++){
        if(arr[i] > *max){
            *max = arr[i];
        }else if(arr[i] <min){
            min = arr[i];
        }
    }
    return min;
}