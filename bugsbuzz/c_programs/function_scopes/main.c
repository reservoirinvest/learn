#include <stdio.h>
void f(int, int, int);

int main(void) {
    int a = 9, b=-2, c=5;
    f(a, b, c);
    printf("a = %d, b = %d, c = %d\n", a, b, c);
    return 0;
}

void f(int x, int y, int z) {
    x = 3+y;
    y = 10*x;
    z = x+y+z;
    printf("x = %d, y = %d, z = %d\n", x,y,z);
}
