# HW3 makefile

CC=nvcc

all: q1 q2

q1: q1.cu
	$(CC) -o q1.out q1.cu

q2: q2.cu
	$(CC) -o q2.out q2.cu

run:
	sbatch q1-batch
	sbatch q2-batch

clean:
	rm -f q[1,2].out q[1,2][a,b,c].txt hw3q[1,2]_*

