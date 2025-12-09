   This is simple pattern recognition application written in python, it can  detect the digits "1", "2", "3" and "0"

   It applies Convolution filters followed by max pooling and stores the results in a postgres sql database.
   A postgres sql running container is mandatory, for that a Dockerfile is provided in pg-sql-docker.
   For evaluation it combines the euclidian distance computation and the cosine prediction

   Overall confidence results for digit "1", "2", "3" and "0" recognition is now 72% but what matters
the most is the training set vs test set and the kernels applied

   It is highly recommended to set-up a python virtual environment for: tensorflow, psycopg2 and PIL and numpy are used for implementation.

Links:
https://www.tigerdata.com/learn/implementing-cosine-similarity-in-python
https://medium.com/@joshuaanang783/what-makes-cnns-so-special-exploring-the-true-nature-of-images-and-how-they-work-with-cnns-36adc103c4be
