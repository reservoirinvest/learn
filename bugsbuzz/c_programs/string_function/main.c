// Unit16_StringFunctions.c
#include <stdio.h>
#include <string.h>
#define MAX_LEN 10

int main(void) {
	char s1[MAX_LEN + 1], s2[MAX_LEN + 1], *p;
	int len;

	printf("Enter string (at most %d characters) for s1: ", MAX_LEN);
	fgets(s1, MAX_LEN+1, stdin);
	len = strlen(s1);
	if (s1[len - 1] == '\n') s1[len - 1] = '\0';

	printf("Enter string (at most %d characters) for s2: ", MAX_LEN);
	fgets(s2, MAX_LEN+1, stdin);
	len = strlen(s2);
	if (s2[len - 1] == '\n') s2[len - 1] = '\0';

    printf("strcmp(s1,s2) = %d\n", strcmp(s1,s2));
	printf("strncmp(s1,s2,3) = %d\n", strncmp(s1,s2,3));

	p = strstr(s1,s2);
	if (p != NULL)
		printf("strstr(s1,s2) returns %s\n", p);
	else
		printf("strstr(s1,s2) returns NULL\n");

	strcpy(s1,s2);
	printf("After strcpy(s1,s2), s1 = %s\n", s1);

	return 0;
}

