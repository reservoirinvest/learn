# Objective
Find the determinant of a square matrix

# What should be done?
Determinant of a square matrix is useful to find an inverse of a matrix.
Determinant is denoted as det A or |A| and is computed as:

[a b      = ad - bc
 c d]

For a 3 x 3 matrix it is:
---

[a b c          = a [e f     -b [d f      +c [d e
 d e f               h i]        g i]         g h]
 g h i]

 = a(ei - fh) - b(di - fg) + c(dh-eg)

For a 4 x 4 matrix it is:
---

 [a b c d      =a [e f g     -b[d f g     +c [d e g     -d [d e f
  d e f g          i j k        h j k         h i k         h i j
  h i j k          m n o]       l n o]        l m o]        l m n]
  l m n o]

  ...this needs to further reduce to:

  = a * (e*[jo-kn] - f[io-km] + g[in-jm]) - b * (d[jo-kn] - f[ho-kl] ...)

Write a recursive function to find the determinant as follows.

int determinant (int mtx[][10], int size)

You may assume size is an integer from 2 to 10.

# How will you do this?

* If size is 2, do the diagonal computation once
* If size is 3, do recursion thrice and diagonal computation once of each recursive matrix
* If size is 4, do recursion 4 times, reduce the matrix to 2 times, and do the diagonal computation
...