//Find the minimum possible difference between elements of an array

#include <stdio.h>
#include <stdlib.h>


int main()
{
    int sz, i, j, arr_diff;
    int min=0;

    printf("Give me the size of your 1-D array\n");
    scanf("%d", &sz);

    int arr[sz];

    //Create the array
    for(i=0; i<sz; i++){
        printf("Enter array value# %d\n", i+1);
        scanf("%d", &arr[i]);
    }

    //Initialize min to get first difference
    min = abs(arr[0] - arr[1]);

    //Traverse through the array to find out the minimum difference
    for(i=0; i<sz; i++){
        for(j=i+1; j<sz; j++){
            arr_diff = abs(arr[i] - arr[j]);
            if (min > arr_diff) min = arr_diff;
        }
    }

    printf("Minimum possible difference is %d\n", min);

    return 0;
}
