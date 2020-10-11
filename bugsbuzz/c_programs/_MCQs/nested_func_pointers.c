#include <stdio.h>
void f(int *a, int *b, int c);
void g(int *a, int *b, int c);

int main(void){
    int a=1, b=2, c=3;
    f(&a, &b, c);
    printf("%d %d %d\n", a, b, c);
    return 0;
}

void f(int *a, int *b, int c){
    g(b, &c, *a);
    g(&c, a, *b);
}

void g(int *x, int *y, int z){
    *y += *x + z;
    z = *x + *y;
}