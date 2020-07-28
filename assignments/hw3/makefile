# HW3 makefile

CC=nvcc

# Build all binaries
all: q1 q2

# Build q1
q1: q1.cu
	$(CC) -o q1.out q1.cu

# Build q2
q2: q2.cu
	$(CC) -o q2.out q2.cu

# Run q1
runq1:
	sbatch q1-batch

# Run q2
runq2:
	sbatch q2-batch

# Run all binaries
run: runq1 runq2

# Clean job output files
cleanjobs:
	rm -f hw3q[1,2]_*

# Clean out all non-version controlled files
clean:
	rm -f q[1,2].out q[1,2][a,b,c].txt hw3q[1,2]_*

