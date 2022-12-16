# The problem

SIC has an array of data pipeline (called A with a size of N)

Based on an a starting index (i) to a ending (d), 
values in the array are to be updated such that,
the updated value is at least or greater than its previous neighbour

[7, 3, 8, 5, 7, 10, 9, 12, 6, 5]

i=3, d=7
[7, 3, 8, 5, 7, 10, 9, 12, 6, 5]
          i                   i+d
          


maxpush(a, N, i, d){
    if a[i+1] > a[i]:
       do nothing
    
    else:
    
    if a[i+1] < a[i]:
        a[i+1] = a[i]
        
    i++ (d times)

}


maxpush(a, N, i, d){
    if a[i+1] < a[i]:
        a[i+1] = a[i]

    i++ (d times)
}