# The question
* Madam Tan has collected a set of coupons.
* Coupons allow to buy certain items at certain price.
   * 3 boxes of chocolates at $10.
* Can only use each coupon once.
* No of items purchased should be exactly same as the number on coupons.

She wants to know 
* minimum cost of buying
* n units of an item
* using only coupons

## Example
If she has 4 coupons ...
1. 3 @ $10
2. 2 @ $4
3. 2 @ $4
4. 1 @ $3

She wants to buy 4 units at minimum cost of $8.

# To-do
Write a recursive function:
int mincost(coupon_t coupons[], int numcoupons, int units)  //to calculate minimum costs

Structure type definition of coupon_t:

typedef struct {
    int quantity;    // No of items purchasable with this coupon
    int price;       // Total price of using the coupon
} coupon_t;

- Return minimum cost if possible, or -1 if not possible.
