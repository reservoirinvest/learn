/*Write a function countNonSpace(char str[])
..to count without using strlen()*/


#include <stdio.h>
#include <string.h>
#include <ctype.h>

int countNonSpace(char str[]);

int main()
{
    char a[20];
    int count;

    printf("Enter a string: \n");
    fgets(a,20,stdin);

    count = countNonSpace(a);
    printf("Count of non-space: %d\n",count);

    return 0;
}

int countNonSpace(char str[]){

    int i, count = 0;

    for (i=0; i<20; i++){
        if (str[i] == '\n'){
            break;
        }
        if (str[i] != ' '){
            count++;
            printf("%c", str[i]);
        }
    }

    return count;
}

  /*char *p = str;
        while(*p++){
            if (*p != ' ')
            count++;

    }
*/

