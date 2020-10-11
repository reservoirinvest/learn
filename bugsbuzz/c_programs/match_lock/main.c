#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int match(char[], char[]);

void main() {
    char pat[20] = {"3E4A1K"};
    char str[20] = {"AEAEAEAK"};
    // char str[20] = {"CAAAAEEEK"};
    // char str[20] ={"AAAEEEK"};

    int result = match(str, pat);

    if (result == 0) {
        printf("OOPS ... NA NA NA NAA NA\n");
    } else
        printf("PASSWORD UNLOCKED !!!\n");
}

int match(char str[], char pat[]) {
    int i, n, j, p, q, count;

    for (i = 0; i < strlen(pat); i += 2) {  // Every 2nd character in pat
        n = pat[i] - '0';
        count = 0;
        for (j = 0; j < strlen(str); j++) {
            if (pat[i + 1] == str[j]) {
                count++;
            }
        }
        if (count != n) {
            return 0;
        }
    }

    // Check if any str is not in pat
    for (q = 0; q < strlen(str); q++) {
        int s = 0;
        for (p = 1; p < strlen(pat); p += 2) {
            if (str[q] == pat[p]) s++;
        }
        if (s == 0) return 0;
    }
    return 1;
}
