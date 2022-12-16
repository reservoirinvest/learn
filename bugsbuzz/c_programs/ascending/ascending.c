#include <stdio.h>

int isAscending(int, int, int);

int main()
{
    int x, y, z, result;

    printf("Enter 3 positive integers:\n");
    scanf("%d %d %d", &x, &y, &z);

    printf("The integers are %d, %d, %d\n", x, y, z);

    result = isAscending(x, y, z);
    if (result == 1)
    {
        printf("Yes. They are ascending.\n");
    }
    else
    {
        printf("No. They are not ascending.\n");
    }

    return 0;
}

int isAscending(int x, int y, int z)
{

    if ((z > y) && (y > x))
    {
        return 1;
    }
    else
    {
        return 0;
    }
}
