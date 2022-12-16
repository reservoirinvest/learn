// Unit16_StringPointer.c
#include <stdio.h>
#include <string.h>

int main(void) {
	char name[12] = "Chan Tan";
	char *namePtr = "Chan Tan";

	printf("name = %s\n", name);
	printf("namePtr = %s\n", namePtr);
	printf("Address of 1st array location for name = %p\n", name);
	printf("Address of 1st array location for namePtr = %p\n", namePtr);

	strcpy(name, "Lee Hsu");
	namePtr = "Lee Hsu";

	printf("name = %s\n", name);
	printf("namePtr = %s\n", namePtr);
	printf("Address of 1st array location for name = %p\n", name);
	printf("Address of 1st array location for namePtr = %p\n", namePtr);

	return 0;
}
