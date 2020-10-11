#include <stdio.h>
#include <string.h>
#include <ctype.h>

void translate(char [], char []);

int main (void){

    char sentence[100];
    //for a string with spaces 
    printf("Enter a sentence in English:\n");
    fgets(sentence, 100, stdin);

    translate(sentence);
    
    return 0;
}

void translate(char sentence[], char word[]){

    // Remove \n and replace with \0

    int len = strlen(sentence);
    if (sentence[len-1] == '\n'){
        sentence[len-1] = '\0';
    }
    
    // Break the sentence into words and store in word[][20]
    // if the character was not a space, save it in word[r][c] and increment c(c++) to store each character of the word 
    //if space is encounterd (end of word), terminate that array with \0 and move to next row (r++) where c=0
    //count++ to keep track of number of spaces = no. of words/array rows 
    //run till end of sentence(encounter \0)
    
    word[20][20] = {{0}};
    int i, j, r=0, c=0, count =0;

    for (i=0; i < len; i++){
        if(sentence[i] != ' '){
            word[r][c] = sentence[i];
            c++;
        }else{
            word[r][c] = '\0';
            r++;
            c=0;
            count++;    //count no. of words 
        }if(sentence[i] == '\0'){
            break;
        }
    }
//replace the first 'a' with 'o'
   for(r=0; r<=count ; r++){
        for(c=0; c < strlen(word[r]); c++){
            if (word[r][c] == 'a'){
                word[r][c] = 'o';
                break;
            }
        }if(r%2 == 0){              //for the first and even row, put wa 
           strcat(word[r], " wa ");
        } else{
            strcat(word[r]," fa " ); // for the odd rows, put fa 
        } 
   }

   for(r=0; r <= count; r++){
       printf("Plutonian: %s", word[r]);
   }

}