
#include <stdio.h>
#define MAX_SIZE 100  // Maximum string size
#define MAX_CHARS 255 // Maximum characters allowed


int main()
{
    char str[MAX_SIZE];
    int freq[MAX_CHARS]; // Store frequency of each character
    int i,a[2], max=0;


    printf("Enter any string: ");
    gets(str);

    /* Initializes frequency of all characters to 0 */
    for(i=0; i<MAX_CHARS; i++)
    {
        freq[i] = 0;
    }


    /* Finds frequency of each characters */

    for (i=0; str[i] != '\0';i++){

        freq[a]++;
        }


    /* Finds maximum frequency */
    for(a=0; i<MAX_CHARS; a++)
    {
        if(freq[i] > freq[max])
            max = i;
    }


    printf("Maximum occurring character is '%c' = %d times.", max, freq[max]);

    return 0;
}

/*#include <stdio.h>

int main(){

    char s[30] = "godisgreater";

    int c[30];

    int len = 12; //manually done

    int i, j;

    for(i=0; i<len; i++){
        c[i] = 1;

        for(j=i+1; j<len; j++){
            if(s[i] == s[j]){
                c[i]++;
                //s[j] = s[j+1];
            }
        }
        //len--;
        printf("%c frequency is %d\n", s[i], c[i]);
    }

    return 0;


}
*/
