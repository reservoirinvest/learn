#include<stdio.h>
#include<string.h>

//Find out where the string breaks!

int main (void){
    char str1[] = {'p', 'e', 'n', '\0', 'c', 'i', 'l', '\0'};
    char str2[14] = {"apple"};
    printf("%s\n", strcat(str2, str1));
    return 0;
}