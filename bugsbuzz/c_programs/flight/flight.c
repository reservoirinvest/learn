#include<stdio.h>

int main(void){

    int s, d;
    int lowest, cost;

    int flight[3][3] = {{0, 60, 45},
                        {40, 0, 26},
                        {31, 22, 0}};

    lowest= flight[0][1]+ flight[1][1];
    for(s=0; s<3; s++){
        for(d=0; d<3; d++){
            if(s!=d){
                cost= flight[s][d]+ flight[d][s];
                if (cost < lowest){
                    lowest = cost;
                    }
                }

                }
            }

    printf("the lowest cost is %d", lowest);
    
    return 0;
}