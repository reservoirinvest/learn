#include<stdio.h>

int main(void){

    int a = 3, *b, c, *d, e, *f;
    
    b = &a; // kachara

    *b = 5; //a = 5 because b first stores address of a and pointer b should point to content of a (replacing 3 to 5)

    c = *b * 3; //c = 5 * 3 = 15

    d = b; //d = address of a, which is kachara

    e = *b + c; // 5 + 15 = 20

    *d = c + e; // 15 + 20 = 35  ... a also gets to be 35, as d = b and *d becomes a. So, a = *b = *d

    f = &e; //address of e which is kachara and *f is 20, which is e

    a = *f + *b; // 20 + 35 = 55. Also *b and *d becomes 55

    *f = *d - *b; // 55 - 55 (as they are equal)

    printf("a = %d, c = %d, e = %d\n", a, c, e);
    printf("*b = %d, *d = %d, *f = %d\n", *b, *d, *f);

    return 0;
}