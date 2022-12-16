#include<stdio.h>

int get_dmt(int [][], int); //determinant function prototype taking matrix and size

int main(void){

    int mtx[4][4] = {{6, 1, 2, 3}, {3, 2, 1, 8}, {1, 1, 2, 2}, {3, 3, 4, 4}};

    int sz = 4; //Matrix size

    int a = reduce(mtx, sz, i);

    return 0;

}

int get_dmt(int a[][10], int sz){
    
    int i, det; //the traverser and the determinant
    int r, c; //row and column

    if ((r > sz) || (c> sz)){ //quit the function
       return a;
    }

    return det = (a[r][c]*a[r+1][c+1] - a[r][c+1] * a[r+1][c]) * (get_dmt(a, sz-))


}
