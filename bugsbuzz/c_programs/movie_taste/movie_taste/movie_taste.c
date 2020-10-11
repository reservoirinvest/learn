#include <stdio.h>
#include <math.h>

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

    int i, j, k, distance = 0, leastdiff = 0, index1, index2, movie;

    for (i = 0; i < n; i++)
    {

        leastdiff = leastdiff + ((ratings[0][i] - ratings[1][i]) *(ratings[0][i] - ratings[1][i]));
    }

    int mindist = sqrt(leastdiff);

    for (i = 0; i < m; i++)
    {
        for (k = i + 1; k < m; k++)
        {
            for (j = 0; j < n; j++)
            {
                movie = movie + ((ratings[i][j] - ratings[k][j])*(ratings[i][j] - ratings[k][j]));
            }

            distance = sqrt(movie);
            if (distance <= mindist)
            {
                mindist = distance;
                index1 = i;
                index2 = k;
            }
        }
    }

    printf("the luck participants are %d %d with score of %d\n", index1, index2, mindist);
}
