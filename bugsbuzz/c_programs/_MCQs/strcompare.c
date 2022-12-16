#include<stdio.h>
#include<string.h>

int main (){
    char *str1 = "Hello", *str2 = "Hello";
    if(str1 == str2)
        printf("equal\n");
    else 
        printf("not equal\n");

    if(strcmp(str1, str2)) //Evaluates to 0 and not 1 because.
         printf("equal\n");
    else //Prints 'not equal' because the value is 0
        printf("not equal\n");

    return 0;
}