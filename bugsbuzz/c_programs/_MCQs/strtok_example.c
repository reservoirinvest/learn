/* strtok example */
#include <stdio.h>
#include <string.h>

void copynwords(char src[], char dest[100], int n);

int main() {
    char src[] = "Be the change you wish to see in the world";
    char dest[100];
    int n = 3;

    copynwords(src, dest, n);

    return 0;
}

void copynwords(char src[], char dest[100], int n) {
    dest[0] = '\0';  // Initialize the destination array by removing kachara

    char *p = strtok(src, " ");  // Break (or) Tokenize src by delimiting space

    while (p != NULL && n > 0) {
        strcat(dest, p);    // Put contents of p in dest
        strcat(dest, " ");  // Put the new space delimiter
        p = strtok(
            NULL,
            " ");  // Move the pointer to the next word after delimiter (space)
        n--;       // Decrement the total number of words needed
    }

    dest[strlen(dest) - 1] = '\0';  // Plug the desitnation's end with a NULL

    for (n = 0; n < strlen(dest); n++) {  // Print the destination string
        printf("%c", dest[n]);
    }
}