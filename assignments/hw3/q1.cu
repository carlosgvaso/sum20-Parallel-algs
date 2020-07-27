/** Homework 3 question 1 code
 *
 * \file q1.cu
 * \author Jose Carlos Martinez Garcia-Vaso <carlosgvaso@utexas.edu>
 * \author Utkarsh Vardan <uvardan@utexas.edu>
 */


#include <cstdio>			// standard I/O
#include <string>			// strings
#include <fstream>			// streams
#include <vector>			// std::vector
#include <sstream>			// std::stringstream
#include <cuda_runtime.h>	// CUDA functions


// Globals
#define DEBUG 1	//! Enable debug messages (0: no messages, 1: some messages, 2: all messages)

#define INPUT_FILE "inp.txt"		//! Input filename
#define OUTPUT_FILE_Q1A "q1a.txt"	//! Q1 a output filename
#define OUTPUT_FILE_Q1B "q1b.txt"	//! Q1 b output filename

#define EXIT_OK 0		//! Exit code success
#define EXIT_FATAL 1	//! Exit code unrecoverable error


/** Read input from file
 *
 * This function assumes the file contains a single line, as per the format in
 * the README.txt.
 *
 * \param	filename	Name of input file to read
 * \return	Vector containing the input array in the file
 */
std::vector<int> read_input (std::string filename) {
	// Create a vector of integers to store the array in file
	std::vector<int> v_in;

	// Create an input filestream
	std::ifstream fin(filename);

	// Make sure the file is open
	if (!fin.is_open()) {
		fprintf(stderr, "ERROR:read_input: Could not open file\n");
		exit(EXIT_FATAL);
	}

	// Helper vars
	std::string line;
	int val;

	// Read the column names
	if (fin.good()) {
		// Extract the first line in the file
		std::getline(fin, line);

		// Create a stringstream from line
		std::stringstream ss(line);

		// Extract each integer
		while (ss >> val) {

			// Add the current integer to the vector
			v_in.push_back(val);

			// If the next token is a comma, ignore it and move on
			if (ss.peek() == ',') ss.ignore();
		}
	}

	// Close file
	fin.close();

	return v_in;
}

/** Write formated output to file
 *
 * This function uses the output format described in the README.txt file.
 *
 * \param	filename	Name of the output file
 * \param	v_out		Vector to save to file
 */
void write_output (std::string filename, const std::vector<int> &v_out) {
	// Create an output filestream object
	std::ofstream fout(filename);

	// Send v_out vector entries to the stream
	for (int i = 0; i < v_out.size(); ++i) {
		fout << v_out.at(i);
		if(i != v_out.size() - 1) fout << ", "; // No comma at end of line
	}
	//fout << "\n";	// inp.txt doesn't have a newline at the end of the file

	// Close the file
	fout.close();
}

/**
 *
 */
__global__ void parallelScanMinKernel(int *d_out, int *d_in, int n)
{
	int gid = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;
	int val = 0;

	// Ensure we only access available array entries
	if (gid < n) {
		#if DEBUG >= 2
		if (tid == 0) {
			printf("\tIterations:\n\t\tBlock %d: d = %d: d_in = [ ",
				   blockIdx.x, 0);
			for (int i=0; i<n; ++i) {
				if (i == n-1) {
					printf("%d ]\n", d_in[i]);
				} else {
					printf("%d, ", d_in[i]);
				}
			}
		}
		#endif
		
		for (int d=1; d<n; d=d*2) {
			if (gid >= d) {
				val = d_in[gid - d];
			}
			__syncthreads();

			if (gid >= d) {
				d_in[gid] = d_in[gid] <= val ? d_in[gid] : val;
			}
			__syncthreads();

			#if DEBUG >= 2
			if (tid == 0) {
				printf("\t\tBlock %d: d = %d: d_in = [ ", blockIdx.x, d);
				for (int i=0; i<n; ++i) {
					if (i == n-1) {
						printf("%d ]\n", d_in[i]);
					} else {
						printf("%d, ", d_in[i]);
					}
				}
			}
			#endif
		}

		/* The result for a block is in the last thread entry for that block.
		 * If n is not a multiple of blockDim.x, the result is the entry of
		 * gid == n-1.
		 */
		if ((tid == blockDim.x-1 && gid != n-1) || gid == n-1) {
			d_out[blockIdx.x] = d_in[gid];

			#if DEBUG
			printf("\t\tBlock %d min: d_out[%d] = %d\n",
				   blockIdx.x, blockIdx.x, d_out[blockIdx.x]);
			#endif
		}
	}
}

/** Q1 a) Compute minA, the minimum value in the input array
 *
 * This function uses the Hillis-Steele version of parallel scan to find the
 * minimum value in the input array. Then, it outputs the result to the
 * OUTPUT_FILE_Q1A output file.
 *
 * This function will only work for problems of size (input array size)
 * (cudaDeviceProp.maxThreadsPerBlock)^2. For example, if we have a
 * cudaDeviceProp.maxThreadsPerBlock = 1024 (a normal value for current Nvidia
 * GPUs), the max problem size is N = 1024^2 = 1,048,576. Since the professor
 * said the max graded size should be 10^6, this restriction sufices.
 *
 * \param	v_in		Input array as a vector
 * \param	dev_props	CUDA device properties
 */
void q1a (const std::vector<int> &v_in, cudaDeviceProp *dev_props) {
	#if DEBUG
	printf("\tTransfering input array to GPU memory...\n");
	#endif

	// Declare GPU memory pointers
	int *d_in, *d_intermediate, *d_out;

	// Allocate GPU memory
	int N = v_in.size();				// Problem size (input array size)
	int d_in_size = N * sizeof(int);	// Input array size in bytes
	int d_out_size = sizeof(int);		// Output array size in bytes

	#if DEBUG
	printf("\tN (input array size): %d\n", N);
	#endif

	if (N > ((int)((*dev_props).maxThreadsPerBlock) * (int)((*dev_props).maxThreadsPerBlock))) {
		fprintf(stderr, "ERROR:q1a: problem size (input array size) is too large\n");
		exit(EXIT_FATAL);
	}

	cudaMalloc((void **) &d_in, d_in_size);
	cudaMalloc((void **) &d_intermediate, d_in_size); // overallocated
	cudaMalloc((void **) &d_out, d_out_size);

	/* Transfer the input array to the GPU
	 * Since the elements of a vector are stored contiguously in memory, we can
	 * pass a pointer to the first element of the vector, and that will act as
	 * if we passed a C array.
	 */
	cudaMemcpy(d_in, &v_in[0], d_in_size, cudaMemcpyHostToDevice);

	#if DEBUG
	// Set up a timer to measure the elapsed time to find the min
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	printf("\tFinding minimum entry in the array...\n");
	cudaEventRecord(start, 0);
	#endif

	// Calculate the number of blocks and threads to use
	int threads_per_block = (int)((*dev_props).maxThreadsPerBlock); // Max number of threads per block
	int blocks_per_grid = (N + (threads_per_block - 1)) / threads_per_block;

	#if DEBUG
	printf("\tThreads per block: %d\n", threads_per_block);
	printf("\tBlocks per grid: %d\n", blocks_per_grid);
	printf("\tRunning kernel...\n");
	#endif

	// Launch the kernel to find min
	parallelScanMinKernel<<<blocks_per_grid, threads_per_block>>>
		(d_intermediate, d_in, N);
	
	// Make sure all the blocks finish executing
	cudaDeviceSynchronize();
	cudaDeviceSynchronize();

	// If there are more than one block, we need to repeat the process with their results
	if (blocks_per_grid > 1) {
		#if DEBUG
		// Copy array to host
		int *a_out;
		a_out = (int*) malloc(d_in_size);
		cudaMemcpy(a_out, d_intermediate, d_in_size, cudaMemcpyDeviceToHost);

		printf("\tBlock results: d_intermediate = [ ");
		for (int i=0; i<blocks_per_grid; ++i) {
			if (i == blocks_per_grid-1) {
				printf("%d ]\n", a_out[i]);
			} else {
				printf("%d, ", a_out[i]);
			}
		}
		free(a_out);
		#endif

		#if DEBUG
		printf("\tThreads per block: %d\n", blocks_per_grid);
		printf("\tBlocks per grid: %d\n", 1);
		printf("\tRunning kernel...\n");
		#endif

		// Fill one block with the results from the other blocks
		parallelScanMinKernel<<<1, blocks_per_grid>>>
			(d_out, d_intermediate, blocks_per_grid);
	}
	
	#if DEBUG
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// Calculate elapsed time, and print it
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("\tAverage time elapsed: %f\n", elapsedTime);
	#endif

	// Copy back the min result from GPU
	int a_out;
	if (blocks_per_grid > 1) {
		cudaMemcpy(&a_out, d_out, d_out_size, cudaMemcpyDeviceToHost);
	} else {
		cudaMemcpy(&a_out, d_intermediate, d_out_size, cudaMemcpyDeviceToHost);
	}

	#if DEBUG
	printf("\tMin: %d\n", a_out);
	#endif
	
	// Copy resulting array to output vector
	std::vector<int> v_out;
	v_out.push_back(a_out);

	// Free GPU memory
	cudaFree(d_in);
	cudaFree(d_intermediate);
	cudaFree(d_out);

	write_output(OUTPUT_FILE_Q1A, v_out);
}

/** Q1 b) Compute an array B such that B[i] is the last digit of A[i] for all i
 *
 * \param	v_in		Input array as a vector
 * \param	dev_props	CUDA device properties
 */
 void q1b (const std::vector<int> &v_in, cudaDeviceProp *dev_props) {
	std::vector<int> v_out;

	// TODO: Implement
	#if DEBUG
	printf("\tThreads per block: %d\n",
		   (int)((*dev_props).maxThreadsPerBlock));
	#endif

	v_out = v_in;

	write_output(OUTPUT_FILE_Q1B, v_out);
}


/** Main
 *
 * Set up CUDA device, read input file, and run Q1a and Q1b.
 *
 * \param	argc	Number of command-line arguments
 * \param	argv	Array of command-line arguments
 * \return	Program return code
 */
int main (int argc, char **argv) {
	#if DEBUG
	std::printf("Executing main...\n");
	#endif

	std::vector<int> v_in;
	int device_count;
	int dev = 0;
	cudaDeviceProp dev_props;

	#if DEBUG
	printf("Detecting CUDA devices...\n");
	#endif

	// Check there are CUDA devices available
	cudaGetDeviceCount(&device_count);
	if (device_count == 0) {
		fprintf(stderr, "ERROR:main: no CUDA devices found\n");
		exit(EXIT_FATAL);
	}

	// Use device 0
	cudaSetDevice(dev);

	if (cudaGetDeviceProperties(&dev_props, dev) == 0) {
		#if DEBUG
		printf("Using device:\n"
			   "\tID: %d\n"
			   "\tName: %s\n"
			   "\tGlobal mem: %d B\n"
			   "\tMax threads per block: %d\n"
			   "\tCompute: v%d.%d\n"
			   "\tClock: %d kHz\n",
			   dev,
			   dev_props.name,
			   (int)dev_props.totalGlobalMem,
			   (int)dev_props.maxThreadsPerBlock,
			   (int)dev_props.major,
			   (int)dev_props.minor,
			   (int)dev_props.clockRate);
		#endif
	} else {
		fprintf(stderr, "ERROR:main: could not find CUDA device information\n");
		exit(EXIT_FATAL);
	}

	#if DEBUG
	std::printf("Reading input array...\n");
	#endif

	// Read input array
	v_in = read_input(INPUT_FILE);

	#if DEBUG >= 2
	printf("\tInput array = [ ");
	for (int i=0; i<v_in.size(); ++i) {
		if (i == v_in.size()-1) {
			printf("%d ]\n", v_in[i]);
		} else {
			printf("%d, ", v_in[i]);
		}
	}
	#endif

	#if DEBUG
	std::printf("Running Q1 a...\n");
	#endif

	// Problem q1 a
	q1a(v_in, &dev_props);

	#if DEBUG
	std::printf("Running Q1 b...\n");
	#endif

	// Problem q1 b
	q1b(v_in, &dev_props);

	#if DEBUG
	std::printf("Done\n");
	#endif

	return 0;
}
