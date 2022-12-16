# include <stdio.h>
int count_numbers(int div1, int div2, int lo, int hi);

void main(){

    int div1, div2, lo, hi;

    while(1)
    {
        printf("Enter 2 divisors:\n");
        scanf("%d %d", &div1, &div2);

        //Check for divisor conditions
        if ((div1 != div2) && div1 > 0 && div2 > 0){
            break;
        }
    }  

    while(1)
    {
        printf("Enter lower and upper limits:\n");
        scanf("%d %d", &lo, &hi);

        //Check for limit conditions
        if (lo > 0 && hi > 0 && hi > lo){
            break;
        }

    }

    printf("Answer = %d\n", count_numbers(div1, div2, lo, hi));

}

int count_numbers(int div1, int div2, int lo, int hi){

    int i, count=0;

    for(i=lo; i<=hi; i++){
        if((i % div1 !=0) && (i % div2 !=0)){
            count++; 
            }
    }

    return count++;

}