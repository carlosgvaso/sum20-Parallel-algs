#!/usr/bin/env python3

# Generate an inp.txt input file with an array of size N with random numbers

from numpy.random import seed
from numpy.random import randint


N = 2000                # Array size
min_int = 0             # Min integer in array
max_int = 999           # Max integer in array
S = 1                   # Seed
filename = "inp.txt"    # Output file

# Seed random number generator
seed(S)

# Generate random numbers between 0-1
arr = randint(min_int, max_int, N)

# Write to file  
with open(filename, 'w') as fout:  
    for i, val in enumerate(arr):
        if i == 0:
            fout.write(str(val))
        else:
            fout.write(', ' + str(val))
