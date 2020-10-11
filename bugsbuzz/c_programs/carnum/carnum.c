#include <stdio.h>
#include <string.h>
#include <ctype.h>

int isValidChecksum(char regNo[]);
char genChecksum(char regNo[]);

int main(void)
{
    char reg[9];
    //assume all inputs are in caps
    printf("Enter car reg. no: ");
    //scan in the reg.no
    fgets(reg, 9, stdin);
    printf("%s\n", reg);

    //replace '/n' with '/0'
    int len = strlen(reg);
    if (reg[len - 1] == '\n')
        reg[len - 1] = '\0';

    //CHECKS TYPE OF CAR AND VALIDITY (0 or 1)
    int sumchk = isValidChecksum(reg);

    //PRINT OUTPUT
    if (sumchk == 0)
    {
        printf("Invalid registration no.\n"); //Invalid
    }
    else
    {
        printf("Valid registration no.\n"); //Valid
    }

    return 0;
}
int isValidChecksum(char reg[])
{
    //IDENTIFY TYPE OF CAR
    switch (reg[1])
    { //Always use break in switch!!!
    case 'H':
        printf("Car is taxi\n");
        break;
    case 'G':
        printf("Car is a goods vehicle\n");
        break;
    case 'Z':
        printf("Car is a rental car\n");
        break;
    default:
        printf("Car is a private car\n");
        break;
    }
    //CHECK VALIDITY OF CHECKSUM
    //Check checksum against last alphabet of reg no reg[(strlen(reg)-1]
    if (genChecksum(reg) == reg[strlen(reg) - 1])
    {
        return 1; //Valid
    }
    else
    {
        return 0; //Invalid
    }
}
//GENERATE CHECKSUM
char genChecksum(char reg[])
{

    char a[27] = {"ABCDEFGHIJKLMNOPQRSTUVWXYZ"}; //Alphabet reference array
    char c[7] = {"945432"};                      //Checksum reference array

    int i, chkd; //remainder of sum-product mod 19

    //REPLACE POSITION 1 AND 2 WITH CORRESPONDING DIGITS
    for (i = 0; i < 27; i++)
    {
        if (reg[1] == a[i])
            reg[1] = i + 1 + '0'; //'+0' converts char to digit
        if (reg[2] == a[i])
            reg[2] = i + 1 + '0';
    }
    printf("2nd character of car is %c\n", reg[1]);
    printf("3rd character of car is %c\n", reg[2]);

    //MODIFIY REG TO ADD PRECEDING ZEROS TO MATCH IDEAL REGISTRATION (8 chars)
    int len = strlen(reg);
    int leadzeros = 8 - len;

    //Move array down to create spaces for leading zeros.
    //Always decrement from the back from NULL
    for (i = 8; i >= 3 + leadzeros; i--)
    {
        reg[i] = reg[i - leadzeros];
    }

    //Fill up the leading zero places with zeros
    //Again... decrement from where the previous loop left
    for (i >= 3 + leadzeros; i > 2; i--)
    {
        reg[i] = '0';
    }

    //GENERATE CHECKSUM

    int sum = 0;

    //Convert the characters to integers by -'0'
    //Sum Product against the mask c
    //Use the boundaries of i to avoid first and last letters of reg
    for (i = 1; i < strlen(reg) - 1; i++)
    {
        sum += (reg[i] - '0') * (c[i - 1] - '0');
    }

    chkd = sum % 19;

    //GENERATE CHECKSUM LETTER
    switch (chkd)
    {
    case 0:
        return 'A';
    case 1:
        return 'Z';
    case 2:
        return 'Y';
    case 3:
        return 'X';
    case 4:
        return 'U';
    case 5:
        return 'T';
    case 6:
        return 'S';
    case 7:
        return 'R';
    case 8:
        return 'P';
    case 9:
        return 'M';
    case 10:
        return 'L';
    case 11:
        return 'K';
    case 12:
        return 'J';
    case 13:
        return 'H';
    case 14:
        return 'G';
    case 15:
        return 'E';
    case 16:
        return 'D';
    case 17:
        return 'C';
    case 18:
        return 'B';
    default:
        return '0';
    }
    //return checksum
}
