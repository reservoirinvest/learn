#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

int main()
{
    int r = 6, c = 8; //total rows and columns
    int i, j, k; //i traverses rows, j traverses columns
    //map the matrix
    int map[6][8] = {
            {25, 58, 53, 23, 21, 34, 21, 50},
            {32, 45, 43, 40, 41, 32, 30, 27},
            {34, 40, 39, 39, 39, 28, 30, 35},
            {40 ,39 ,42 ,48 ,39 ,34 ,29 ,32},
            {39, 39, 39, 39, 39, 49, 27, 30},
            {31, 31, 31, 32, 32, 33, 44, 35}
            };

    int offset[8][2] = {
    {-1, -1},
    {-1, 0},
    {-1, 1},
    {0, -1},
    {0, 1},
    {1, -1},
    {1, 0},
    {1, 1}
    };

    int hikeoffset[4][2] = {
    {0, 1}, //always check right and down
    {1, 0},
    {-1, 0},
    {0, -1}
    };

    int pvmap[6][8] = {{0}};
    int hike[6][8] = {{0}};

    //Check for Peaks
    for (i=1; i<r-1; i++){ //Row traverse - starts
        for(j=1; j<c-1; j++) { //Column traverse
            int pv = 0; //reset pv each time a shift in i, j occurs in the map
            for(k=0; k<8; k++){ //Looping through offset
                if(map[i][j] > map[i+offset[k][0]][j+offset[k][1]]){
                    pv++;
                } else {
                    pv--;
                }
            }
            if(pv==8) {
                printf("%d at [%d][%d] is a peak!!\n", map[i][j], i, j);
                pvmap[i][j] = 1;
            } else if (pv==-8) {
                printf("%d at [%d][%d] is a valley!!\n", map[i][j], i, j);
                pvmap[i][j] = 2;
            } else pvmap[i][j] = 0;
        }
    }

    for (i=1; i<r-1; i++){ //Row traverse - starts
        for(j=1; j<c-1; j++) { //Column traverse
            int route = 0; //reset pv each time a shift in i, j occurs in the map
            for(k=0; k<8; k++){ //Looping through offset
                if(map[i][j] - (map[i+hikeoffset[k][0]][j+hikeoffset[k][1]])<=2){
                    pv++;
                } else {
                    pv--;
                }
            }
            if(pv==8) {
                printf("%d at [%d][%d] is a peak!!\n", map[i][j], i, j);
                pvmap[i][j] = 1;
            } else if (pv==-8) {
                printf("%d at [%d][%d] is a valley!!\n", map[i][j], i, j);
                pvmap[i][j] = 2;
            } else pvmap[i][j] = 0;
        }
    }
    return 0;
}
