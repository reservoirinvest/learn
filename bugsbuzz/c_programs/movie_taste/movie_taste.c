#include <stdio.h>
#include <math.h>
#include <stdlib.h>

void pairing(int[4][5], int, int);

int main(void)
{
    int m = 4; // No of participants
    int n = 5; // No of movies

    int ratings[4][5] = {{4, 4, 3, 3, 2}, {4, 3, 1, 1, 5}, {5, 5, 5, 2, 2}, {4, 3, 1, 1, 4}};

    pairing(ratings, m, n);

    return 0;
}

void pairing(int ratings[4][5], int m, int n){

    int index1, index2;
    int mindiff, i, j, k, p, q, diff, sum=0, total = 0;

    //...Initialize mindiff

    for (k = 0; k < n; k++) {
        q = pow((ratings[0][k] - ratings[1][k]), 2);
        total = total + p;
    }

    mindiff = sqrt(total);

    //... Calculate the minimum distance
    for (i = 0; i < m; i++) {
        for (j = i + 1; j < m; j++) {
            sum = 0;
            for (k = 0; k < n; k++) {
                p = pow((ratings[i][k] - ratings[j][k]), 2);
                sum = sum + p;
            }
            diff = sqrt(sum);

            if (diff < mindiff) {
                mindiff = diff;
                index1 = i;
                index2 = j;
            }
        }
    }

    printf("the luck participants are %d %d with score of %d\n", 
            index1, index2, mindiff);
}
