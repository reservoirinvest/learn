#include<stdio.h>

typedef struct {
    int quantity;    // No of items purchasable with this coupon
    int price;       // Total price of using the coupon
} coupon_t;

int mincost(coupon_t coupons[], int numcoupons, int units);
int readinputs(coupon_t coupons[])

int main(void){
    coupon_t coupons[20];

    int numcoupons = readinput(coupons);

    int units;
    printf("Enter the number of units to be purchased: \n");
    scanf("%d", &units);

    int min = mincost(coupons, numcoupons, units);

    return 0;
}

}
int readinputs(coupon_t coupons[]){

    int numcoupons, int i;
    printf("Enter the number of coupons in list\n");
    scanf("%d", &coupons);

    for(i=0; i<numcoupons, i++){
        scanf("%d %d", &coupons[i].quantity, coupons[i].price);
    }

    return numcoupons;
}

int mincost(coupon_t coupons[], int numcoupons, int units){
max = coupons[0].quantity
for(i=0; i<numcupons; i++){
    if(max<coupon[i].quantity){
        max = coupon[i].quantity;
    }
    
    if(units != )
}

    


}

