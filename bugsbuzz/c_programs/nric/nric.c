#include <stdio.h>
#include <math.h>

char generateCode(int);
int main(void)
{

    int n, i, d, dw, result, sum=0;
    //char code;
    int weight = 2765432;

    printf("Enter your NRIC: \n");
    scanf("%d", & n);

    for(i=7; i>=0; i--) {
        d = n/pow(10,i-1);
        n =  n % (int) pow(10,i-1);

        printf("digit is: %d\n", d);
        printf("new n is: %d\n\n", n);

        dw = weight/pow(10, i-1);
        weight = weight % (int) pow(10,i-1);

        printf("weight digit is: %d\n", dw);
        printf("new weight is: %d\n\n", weight);

        sum = sum + d*dw;

        printf("sum is: %d\n\n", sum);
        result = 11 - (sum % 11);
        printf("result is: %d\n", result);

    }
    printf("nric check code is: %c\n", generateCode(result));
    return 0;
}
    char generateCode(int result){
        switch(result){
            case 1: return 'A';
            case 2: return 'B';
            case 3: return 'C';
            case 4: return 'D';
            case 5: return 'E';
            case 6: return 'F';
            case 7: return 'G';
            case 8: return 'H';
            case 9: return 'I';
            case 10: return 'Z';
            case 11: return 'J';
            default: return "Unknown";
        }
    }
