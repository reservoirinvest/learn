#include <stdio.h>
#include <math.h>

/*
Write a program magicNumber.c to read two positive integers, each with at most 5 digits, and
for each integer, add up the digits (from right) in positions 1, 3 and 5. The right-most digit of the
sum is the required answer.

For example, if the input is 76524, then adding up the digits 4, 5 and 7, we get 16. The answer is
hence 6.

Your program should contain a function get_magic() to compute and return the answer. You
are to decide on its parameter(s). What is the precondition of the function?

Sample run:
===========
Enter 1st value: 76524
Magic number = 6

Enter 2nd value: 8946
Magic number = 5
*/

/* To-do:
=========
1) Check input for 5 digits
2) Check input for +ve nos and no alphabets
3) Make it for 2 numbers
*/

int get_len(int);

int main(void){
	int n, count, sum=0;
	printf("Enter an integer upto 5 digits: \n");
	scanf("%d", &n);

	count = get_len(n);

	printf("The length of input %d is %d\n", n, count);

	//For odd length numbers
	if (count % 2 == 1) {
        do {
            sum = sum + n % 10;
            n = n/100;
        } while (n>0);
	}

	//For even length numbers
	else {
        do {
            sum = sum + ((n/10) % 10);
            n = n/100;
        } while (n>0);
	}

	printf("The magic sum is: %d\n", sum);
	printf("The magic number is: %d\n", sum % 10);


	return 0;
}

//Determine length of the number
int get_len(int n){
	int count = 0;

	while (n !=0) {
		n = n/10;
		count++;
	}

	return count;
}
