This is simple pattern recognition application written in python.

It uses Convolution followed by max pooling and compare the results with similar data
from the postgresal databse. It uses euclidian distance to evaluate the possible resemblance.

For training a Postgres SQL database is nedeeed, a Dockerfile is provided in pg-sql-docker 
directory, one_images/ two_images/ and houses/ contains associated images used for training.

It is highly recommended to set-up a python virtual environment, tensorflow, PIL qnd numpy are used for implementation.
