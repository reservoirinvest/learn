# Problem
25 treasures are arranged in a matrix.
Different value at each co-ordinate.

```
1	3	3	1	1
1	6	1	2	1
2	1	8	1	5
1	1	8	1	5
1	4	1	1	4
```


## First assignment
Sum values of the squares within coordinates given.

## Second assignment
Take the array. Integer representing size of area.
Return coordinates of most valuable area of the size.

# Pseudocode

## First assignment
Determine the start coordinate. (i, j)
Use the size to get the end coordinates. (p, q)
Build a loop to traverse through the limits of (i,j) -> (p, q)
Do a cumulative sum to get the treasure value
Do a count++ to get the size of area covered

## Second assignment
Start from (0,)