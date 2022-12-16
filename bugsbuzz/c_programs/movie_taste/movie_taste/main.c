#include <stdio.h>
#include <math.h>
#include <stdlib.h>

void most_similar(int[4][5], int, int);

int main(void)
{
    int m = 4; // No of participants
    int n = 5; // No of movies

    int ratings[4][5] = {{4, 4, 3, 3, 2}, {4, 3, 1, 1, 5}, {5, 5, 5, 2, 2}, {4, 3, 1, 1, 4}};

    most_similar(ratings, m, n);

    return 0;
}

void most_similar(int ratings[4][5], int m, int n)
{

    int i, j, k, distance = 0, leastdiff = 0, index1, index2, movie = 0;

    for (i = 0; i < n; i++)
    {

        leastdiff = leastdiff + (int) pow((ratings[0][i] - ratings[1][i]), 2);
    }

    int mindist = sqrt(leastdiff);
    int arr[4] = {0};

    for (i = 0; i < m; i++)
    {
        for (k = i+1; k < m; k++){
            for (j = 0; j < n; j++){
                movie = movie + (int) pow((ratings[i][j] - ratings[k][j]), 2);
            }
            distance = sqrt(movie);

            if (distance <= mindist){
                mindist = distance;
                index1 = i;
                index2 = k;
                movie = 0;
            }
        }
        arr[i] = distance;
    }

    printf("the luck participants are %d %d with score of %d\n", index1, index2, mindist);
}
