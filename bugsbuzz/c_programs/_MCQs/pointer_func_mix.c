/* Waiting for a response from Parineeta.
   The value of c differs between compiler output and hand output.
   Hand output = 2.
   Compiler output = 0.
*/

#include<stdio.h>

void g(int *, int *, int *);


int main(void){

    int a = 1, b = 2, c = 3;

    g(&c, &a, &b);

    printf("%d %d %d\n", a, b, c);

    return 0;
}

void g(int *x, int*y, int *z){
    *y = *z/(*x)--;
}
