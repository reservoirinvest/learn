# Objective
Write a recursive function to fill pattern as follows:

{1}

{1, 2}, {2, 2}

{1, 2, 3}, {2, 2, 3}, {3, 3, 3}

{1, 2, 3, 4}, {2, 2, 3, 4}, {3, 3, 3, 4}, {4, 4, 4, 4}

...

# Pseudocode
Establish a 20 x 20 array
Start with row 1, column 1
Expand it to 2 rows and 2 columns and fill 2s
Expand it to 3 rows and 3 columns from the 2 rows and 2 columns and fill 3s
Expand it to 4 rows and 4 columns from the 3 rows and 3 columns and fill 4s 


