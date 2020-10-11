# The scenario
There are m participants
Are asked to gave rating of 1 - 5 *s for n movies each

Company decides to give a pair of movie tickets to two participants with most similar taste.

Use Euclidian distance as a measure of similarity
distance = sqrt[(ri1 - rj1)sq + (ri2 - rj2)sq + ... + (rin - rjn)sq]
Participants with the smallest Euclidian distance wins the tickets

# The challenge
Write a function _most similar_ taking 2D array and returns the indices of two participants with most similar movie taste.
At least 2 partipants
Movies (1... 100)

# Pseduocode

## Entry
There is a ratings bucket. 
This bucket has name of participant and his rating for each movie.

Participant 0 enters rating for 0 to n movies
Participant 1 enters rating for 0 to n movies
...

Participant m enters rating for 0 to n movies

## Processing
We need to find the closes match. This is done by 'pairing'

For the first participant (0) select another participant (0+1).
   Get their difference in rating for movie 0. Square it.
   Get their difference in rating for movie 2. Square it.

    ...

   Get their difference in rating for movie n. Square it.

   Sum all their differences for 0...n movies

   Square-root the sum and store it in 'mindiff'.
   Store indexes of participant (m) and (m+1)


    Select the next participant (m+2) to compare with the first (m) and repeat to get a 'square-root'

    If this square-root is < mindiff, 
       ...replace mindiff with the square-root
       ...replace indexes with participant (m) and (m+2)

Repeat the entire process till you all participants are exhausted with everyone else in the pool (till m is reached)

The last remaining mindiff is the smallest value
The last remining index is the ideal pair
