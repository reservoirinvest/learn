#include <stdio.h>
#include <string.h>
#include <ctype.h>

void convertString(char str[], char dest[]);

int main()
{
    char str[21], dest[100];

    printf("Enter a string: \n");
    fgets(str,21,stdin);
    convertString(str, dest);

    return 0;
}

void convertString(char str[], char dest[]){

    int i, j = 0;

    for(i=0; i<= 21; i++){
        if(str[i] != ' '){ //only non-space characters are copied into dest
            dest[j] = str[i];
            dest[j+1] = '*';
            j = j+2; //next available space
        }

    }

    printf("Converted string is %s\n", dest);

}
