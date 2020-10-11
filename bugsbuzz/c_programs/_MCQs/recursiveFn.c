/*
KEY LEARNINGS!!!!
=================

1) When function calls itself in 2 iterations or more, you input values back to the function from right to left.
2) When it does not return a function, the value is simply stored in the stack as n.
3) You have to draw branches (sub-stacks) from main stack and work the result backwards.
4) If a printf is after the function is called, it is accumulated and prints step-by-step as iteration in all the stacks.
*/

#include<stdio.h>
int f(int n);

int main(void){
    int n = 13;
    printf("%d\n", f(n));
    return 0;
}


int f(int n){
    printf("inside f is: %d\n", n);

    switch(n%4){
        case 0: return n;
        case 2: return 2*f(n/4);
        default: return f(n-1)+f(n+1); 
    }
}