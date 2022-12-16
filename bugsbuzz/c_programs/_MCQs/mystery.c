#include<stdio.h>

/* This function shows winding and un-winding recursion
   Mystery calls itself without executing putchar, until n = 0. ...15/3, 5/3, 1/3, 0
   After each execution, a stack is saved in memory, which also includes the unexecuted putchar.
   When it reaches the condition of n = 0, the function starts unwinding to build back the solution ... 0, 1/3, 5/3, 15/3.
   This eliminates the need for a 'return' in recursion.
*/
void mystery(int n, int b);


int main(void){

    int n = 15, d = 3;
    mystery(n, d);

    return 0;
}

void mystery(int n, int b){
    if(n != 0) {
        mystery(n/b, b);
        putchar('b'+ n%b);
    }
}
