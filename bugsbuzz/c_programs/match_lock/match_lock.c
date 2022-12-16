#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int match(char[], char[]);

void main() {
    char pat[20] = {"3E4A1K"};
    // char str[20] = {"AEAEAEAK"}; //Successful one
    // char str[20] = {"CAAAAEEEK"};
    char str[20] ={"AAAEEEK"};

    int result = match(str, pat);

    if (result == 0) {
        printf("OOPS ... NA NA NA NAA NA\n");
    } else
        printf("PASSWORD UNLOCKED !!!\n");
}

int match(char str[], char pat[]) {
    int i, n, j, p, q, count;

    //... Find out pattern not in string

    for (i = 0; i < strlen(pat); i += 2) {  //.. Every 2nd character in pat
        n = pat[i] - '0'; //.. Convert character to integer
        count = 0; //.Resets with every iteration
        for (j = 0; j < strlen(str); j++) {
            if (pat[i + 1] == str[j]) {
                count++;
            }
        }
        //.. Quantity of matched characters should be EXACT!
        if (count != n) {
            return 0;
        }
    }

    //... Find out string not in pattern (e.g. extra letters)
    for (q = 0; q < strlen(str); q++) {
        int s = 0;
        for (p = 1; p < strlen(pat); p += 2) {
            if (str[q] == pat[p]) s++;
        }
        if (s == 0) return 0; //..Breaks out if even one character is not present in pattern
    }

    //..Do the above BEFORE the final return
    return 1;
}
