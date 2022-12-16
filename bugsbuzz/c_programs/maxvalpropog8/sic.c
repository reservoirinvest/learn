#include<stdio.h>

void maxpush(int[], int, int, int);
void maxpull(int[], int, int, int);

int main(void){

    int a[10] = {11, 3, 5, 8, 19, 7, 10, 12, 6, 9};
    int i = 0;  //start recursion from position
    int d = 4;  //times to do recursion
    int n = 10; //size of array
    int m = i+d-1; //make changes from a[i] to a[i+d-1]

    // maxpush(a, n, i, m);
    maxpull(a,n,i,m);

    return 0;
}
//updated segment has at least the same/or more capacity as the connector before it 
void maxpush( int arr[], int n, int i, int m){
 //printf within the function   
    int j;

    if(i==m){
        for (j=0; j<n; j++){
            printf("%d \n", arr[j]);  //insert printf here so that array is printed before quitting function 
        }
        return;                     //return serves to quit the function once condition is not met 
    }if(arr[i+1] < arr[i]){
        arr[i+1] = arr[i];
        maxpush(arr, n, i+1, m);        //increment i to evaluate the next element 
    } else{
        maxpush(arr, n, i+1, m);        //NOT A FOR LOOP; if you dont re-call the function when there are no changes to be made, the function will just quit. 
    }*
}
//updated segment has at least the same or more capaciy than the connector after it
void maxpull( int arr[], int n, int i, int m){
    int j;
    if (m == i){
        for (j=0; j<n; j++){
              printf("%d ", arr[j]);
        }
        return;                 // return must be outside of if-brackets to execute
    }
    if (arr[m-1] < arr[m]){
        arr[m-1] = arr[m];
    }
    maxpull(arr, n, i, m-1);    //to avoid if-else repetition (see maxpush), put function ouside of if so whichever scenario executed is stored in array. 
}