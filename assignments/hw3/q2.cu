/** Homework 3 question 2 code
 *
 * \file q2.cu
 * \author Utkarsh Vardan <uvardan@utexas.edu>
 * \author Jose Carlos Martinez Garcia-Vaso <carlosgvaso@utexas.edu>
 */


#include <cstdio>			// standard I/O
#include <string>			// strings
#include <fstream>			// streams
#include <vector>			// std::vector
#include <sstream>			// std::stringstream
#include <cuda_runtime.h>	// CUDA functions


// Globals
#define DEBUG 0	//! Enable debug messages (0: no log output, 1: non-verbose logs, 2: verbose logs, 3: all logs)

#define INPUT_FILE "inp.txt"
#define OUTPUT_FILE_Q2A "q2a.txt"
#define OUTPUT_FILE_Q2B "q2b.txt"
#define OUTPUT_FILE_Q2C "q2c.txt"

#define EXIT_OK 0		//! Exit code success
#define EXIT_FATAL 1	//! Exit code unrecoverable error

#define RANGES_NUM 10	//! Number of ranges (d_out size)


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
	std::vector<int> arr_in;

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
			arr_in.push_back(val);

			// If the next token is a comma, ignore it and move on
			if (ss.peek() == ',') ss.ignore();
		}
	}

	// Close file
	fin.close();

	return arr_in;
}

/** Write formated output to file
 *
 * This function uses the output format described in the README.txt file.
 *
 * \param	filename	Name of the output file
 * \param	arr_out		Vector to save to file
 */
void write_output (std::string filename, std::vector<int> arr_out) {
	// Create an output filestream object
	std::ofstream fout(filename);

	// Send arr_out vector entries to the stream
	for (int i = 0; i < arr_out.size(); ++i) {
		fout << arr_out.at(i);
		if(i != arr_out.size() - 1) fout << ", "; // No comma at end of line
	}
	//fout << "\n";	// inp.txt doesn't have a newline at the end of the file

	// Close the file
	fout.close();
}

/** CUDA kernel for counting the entries in parallel using global memory
 *
 * \param	d_out	Pointer to output array in global memory
 * \param	d_in	Pointer to input array in global memory
 * \param	n		Size of the problem (input array size)
 */
__global__ void counterGlobalKernel(int *d_out, int *d_in, int n) {
	int tid=threadIdx.x;
	int blockid=blockIdx.x;
	int offset= blockid * blockDim.x;
	int gid=tid + offset;

	if (gid < n){
		if (d_in[gid]>=0 && d_in[gid]<100){
			atomicAdd(&d_out[0],1);
		}
		else if (d_in[gid]>=100 && d_in[gid]<200){
			atomicAdd(&d_out[1],1);
		}
		else if (d_in[gid]>=200 && d_in[gid]<300){
			atomicAdd(&d_out[2],1);
		}
		else if (d_in[gid]>=300 && d_in[gid]<400){
			atomicAdd(&d_out[3],1);
		}
		else if (d_in[gid]>=400 && d_in[gid]<500){
			atomicAdd(&d_out[4],1);
		}
		else if (d_in[gid]>=500 && d_in[gid]<600){
			atomicAdd(&d_out[5],1);
		}
		else if (d_in[gid]>=600 && d_in[gid]<700){
			atomicAdd(&d_out[6],1);
		}
		else if (d_in[gid]>=700 && d_in[gid]<800){
			atomicAdd(&d_out[7],1);
		}
		else if (d_in[gid]>=800 && d_in[gid]<900){
			atomicAdd(&d_out[8],1);
		}
		else if (d_in[gid]>=900 && d_in[gid]<1000){
			atomicAdd(&d_out[9],1);
		}
	}
	__syncthreads();

	#if DEBUG >= 2
	if (gid == 0) {
		printf("\t\tResult: [ ");
		for (int i=0; i<RANGES_NUM; ++i) {
			if (i == RANGES_NUM-1) {
				printf("%d ]\n", d_out[i]);
			} else {
				printf("%d, ", d_out[i]);
			}
		}
	}
	#endif
}

/** CUDA kernel for counting the entries in parallel using shared memory
 *
 * \param	d_out	Pointer to output array in global memory
 * \param	d_in	Pointer to input array in global memory
 * \param	n		Size of the problem (input array size)
 */
__global__ void counterSharedKernel(int *d_out, int *d_in, int n) {
	/* d_shared is allocated in the kernel call 3rd arg: <<<blk, th, shmem>>>.
	 * We allocated RANGES_NUM extra entries in the array to save each block's
	 * results to shared memory at the end of the array, after the input arrays
	 * entries.
	 */
	extern __shared__ int d_shared[];

	int tid = threadIdx.x;
	int blockid = blockIdx.x;
	int offset = blockid * blockDim.x;
	int gid = tid + offset;

	// Load shared mem from shared mem
	if (gid < n) {
		d_shared[tid] = d_in[gid];

		// Initialize the results part of the array to all zeroes
		if (tid == 0) {
			for (int i=n; i<n+RANGES_NUM; ++i) {
				d_shared[i] = 0;
			}
		}
	}
	__syncthreads();

	if (gid < n){
		if (d_shared[tid]>=0 && d_shared[tid]<100){
			atomicAdd(&d_shared[n+0],1);
		}
		else if (d_shared[tid]>=100 && d_shared[tid]<200){
			atomicAdd(&d_shared[n+1],1);
		}
		else if (d_shared[tid]>=200 && d_shared[tid]<300){
			atomicAdd(&d_shared[n+2],1);
		}
		else if (d_shared[tid]>=300 && d_shared[tid]<400){
			atomicAdd(&d_shared[n+3],1);
		}
		else if (d_shared[tid]>=400 && d_shared[tid]<500){
			atomicAdd(&d_shared[n+4],1);
		}
		else if (d_shared[tid]>=500 && d_shared[tid]<600){
			atomicAdd(&d_shared[n+5],1);
		}
		else if (d_shared[tid]>=600 && d_shared[tid]<700){
			atomicAdd(&d_shared[n+6],1);
		}
		else if (d_shared[tid]>=700 && d_shared[tid]<800){
			atomicAdd(&d_shared[n+7],1);
		}
		else if (d_shared[tid]>=800 && d_shared[tid]<900){
			atomicAdd(&d_shared[n+8],1);
		}
		else if (d_shared[tid]>=900 && d_shared[tid]<1000){
			atomicAdd(&d_shared[n+9],1);
		}
	}
	__syncthreads();

	// Only 1 thread syncs data to global memory
	if (tid == 0) {
		for (int i=0; i<RANGES_NUM; ++i) {
			atomicAdd(&d_out[i], d_shared[n+i]);
		}

		#if DEBUG >= 2
		printf("\t\tResult: Block %d: [ ", blockIdx.x);
		for (int i=0; i<RANGES_NUM; ++i) {
			if (i == RANGES_NUM-1) {
				printf("%d ]\n", d_shared[n+i]);
			} else {
				printf("%d, ", d_shared[n+i]);
			}
		}
		#endif
	}
}

/** CUDA kernel for the Hillis-Steele parallel scan sum
 *
 * \param	d_in	Pointer to input array in global memory
 * \param	n		Size of the problem (input array size)
 */
__global__ void parallelScanSumKernel(int *d_in, int n) {
	// Initialize global and thread IDs, and other variables
	int gid = threadIdx.x + blockDim.x * blockIdx.x;
	#if DEBUG >= 3
	int tid = threadIdx.x;
	#endif
	int val = 0;

	// Ensure we only access available array entries
	if (gid < n) {
		#if DEBUG >= 3
		if (tid == 0) {
			printf("\t\tIterations:\n\t\t\tBlock %d: d = %d: d_in = [ ",
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
				d_in[gid] = d_in[gid] + val;
			}
			__syncthreads();

			#if DEBUG >= 3
			if (tid == 0) {
				printf("\t\t\tBlock %d: d = %d: d_in = [ ", blockIdx.x, d);
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

		#if DEBUG >= 2
		if (gid == n-1) {
			printf("\t\tResult: [ ");
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
}

/** Q2 a) Compute a counter array in global memory of GPU
 *
 * Create an array B of size 10 that keeps a count of the entries in each of the
 * ranges:[0, 99], [100, 199], [200, 299], ... , [900, 999]. For this part of
 * the problem, maintain array B in global memory of GPU.
 *
 * \param	v_in		Input array as a vector
 * \param	dev_props	CUDA device properties
 * \return	Output vector with the contents of array B
 */
std::vector<int> q2a (const std::vector<int> &v_in, cudaDeviceProp *dev_props) {
	#if DEBUG
	printf("\tTransfering input array to GPU memory...\n");
	#endif

	// Declare GPU memory pointers
	int *d_in, *d_out;

	// Allocate GPU memory
	int N = v_in.size();					// Problem size (input array size)
	int N_out = RANGES_NUM;					// Output array size
	int d_in_size = N * sizeof(int);		// Input array size in bytes
	int d_out_size = N_out * sizeof(int);	// Output array size in bytes

	// Allocate output array, and initilize to all zeroes 
	int *a_out;
	a_out = (int*) calloc(N_out, sizeof(int));

	#if DEBUG
	printf("\tN (input array size): %d\n", N);
	#endif

	/*
	if (N > ((int)((*dev_props).maxThreadsPerBlock) * (int)((*dev_props).maxThreadsPerBlock))) {
		fprintf(stderr, "ERROR:q1a: problem size (input array size) is too large\n");
		exit(EXIT_FATAL);
	}
	*/

	cudaMalloc((void **) &d_in, d_in_size);
	cudaMalloc((void **) &d_out, d_out_size);

	/* Transfer the input and output arrays to the GPU
	 * Since the elements of a vector are stored contiguously in memory, we can
	 * pass a pointer to the first element of the vector, and that will act as
	 * if we passed a C array.
	 */
	cudaMemcpy(d_in, &v_in[0], d_in_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_out, a_out, d_out_size, cudaMemcpyHostToDevice);

	#if DEBUG
	// Set up a timer to measure the elapsed time to find the min
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	printf("\tCounting entries in global memory...\n");
	#endif

	// Calculate the number of blocks and threads to use
	int threads_per_block = (int)((*dev_props).maxThreadsPerBlock); // Max number of threads per block
	int blocks_per_grid = (N + (threads_per_block - 1)) / threads_per_block;

	#if DEBUG
	printf("\tThreads per block: %d\n", threads_per_block);
	printf("\tBlocks per grid: %d\n", blocks_per_grid);
	printf("\tRunning kernel...\n");
	cudaEventRecord(start, 0);
	#endif

	// Launch the kernel to find min
	counterGlobalKernel<<<blocks_per_grid, threads_per_block>>>
		(d_out, d_in, N);
	
	// Make sure all the blocks finish executing
	cudaDeviceSynchronize();
	cudaDeviceSynchronize();
	
	#if DEBUG
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// Calculate elapsed time, and print it
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("\tAverage time elapsed: %f\n", elapsedTime);
	#endif

	// Copy back the result from GPU
	cudaMemcpy(a_out, d_out, d_out_size, cudaMemcpyDeviceToHost);

	#if DEBUG >= 2
	printf("\ta_out = [ ");
	for (int i=0; i<N_out; ++i) {
		if (i == N_out-1) {
			printf("%d ]\n", a_out[i]);
		} else {
			printf("%d, ", a_out[i]);
		}
	}
	#endif
	
	// Copy resulting array to output vector
	std::vector<int> v_out (a_out, a_out + N_out);

	#if DEBUG
	printf("\tOutput = [ ");
	for (int i=0; i<v_out.size(); ++i) {
		if (i == v_out.size()-1) {
			printf("%d ]\n", v_out[i]);
		} else {
			printf("%d, ", v_out[i]);
		}
	}
	#endif


	// Free GPU memory
	cudaFree(d_in);
	cudaFree(d_out);

	// Free host memory
	free(a_out);

	// Save output to file
	write_output(OUTPUT_FILE_Q2A, v_out);

	// Return the output vector to be used in Q2 c
	return v_out;
}

/** Q2 b) Compute a counter array in shared memory of GPU
 *
 * Repeat part (a) but first use the shared memory in a block for updating the
 * local copy of B in each block. Once every block is done, add all local copies
 * to get the global copy of B.
 *
 * \param	v_in		Input array as a vector
 * \param	dev_props	CUDA device properties
 */
void q2b (const std::vector<int> &v_in, cudaDeviceProp *dev_props) {
	#if DEBUG
	printf("\tTransfering input array to GPU memory...\n");
	#endif

	// Declare GPU memory pointers
	int *d_in, *d_out;

	// Allocate GPU memory
	int N = v_in.size();					// Problem size (input array size)
	int N_out = RANGES_NUM;					// Output array size
	int d_in_size = N * sizeof(int);		// Input array size in bytes
	int d_out_size = N_out * sizeof(int);	// Output array size in bytes

	// Allocate output array, and initilize to all zeroes 
	int *a_out;
	a_out = (int*) calloc(N_out, sizeof(int));

	#if DEBUG
	printf("\tN (input array size): %d\n", N);
	#endif

	/*
	if (N > ((int)((*dev_props).maxThreadsPerBlock) * (int)((*dev_props).maxThreadsPerBlock))) {
		fprintf(stderr, "ERROR:q1a: problem size (input array size) is too large\n");
		exit(EXIT_FATAL);
	}
	*/

	cudaMalloc((void **) &d_in, d_in_size);
	cudaMalloc((void **) &d_out, d_out_size);

	/* Transfer the input and output arrays to the GPU
	 * Since the elements of a vector are stored contiguously in memory, we can
	 * pass a pointer to the first element of the vector, and that will act as
	 * if we passed a C array.
	 */
	cudaMemcpy(d_in, &v_in[0], d_in_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_out, a_out, d_out_size, cudaMemcpyHostToDevice);

	#if DEBUG
	// Set up a timer to measure the elapsed time to find the min
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	printf("\tCounting entries in shared memory...\n");
	#endif

	// Calculate the number of blocks and threads to use
	int threads_per_block = (int)((*dev_props).maxThreadsPerBlock); // Max number of threads per block
	int blocks_per_grid = (N + (threads_per_block - 1)) / threads_per_block;

	#if DEBUG
	printf("\tThreads per block: %d\n", threads_per_block);
	printf("\tBlocks per grid: %d\n", blocks_per_grid);
	printf("\tRunning kernel...\n");
	cudaEventRecord(start, 0);
	#endif

	/* Launch the kernel to find min
	 * The 3rd arg to kernel is the size of the shared memory array. This is the
	 * size of the input array plus the size of the output array, because we
	 * save input array and each block's results to shared memory. The input
	 * array is saved first, and the output array is saved after this.
	 */
	counterSharedKernel<<<blocks_per_grid, threads_per_block, d_in_size+d_out_size>>>
		(d_out, d_in, N);
	
	// Make sure all the blocks finish executing
	cudaDeviceSynchronize();
	cudaDeviceSynchronize();
	
	#if DEBUG
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// Calculate elapsed time, and print it
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("\tAverage time elapsed: %f\n", elapsedTime);
	#endif

	// Copy back the result from GPU
	cudaMemcpy(a_out, d_out, d_out_size, cudaMemcpyDeviceToHost);

	#if DEBUG >= 2
	printf("\ta_out = [ ");
	for (int i=0; i<N_out; ++i) {
		if (i == N_out-1) {
			printf("%d ]\n", a_out[i]);
		} else {
			printf("%d, ", a_out[i]);
		}
	}
	#endif
	
	// Copy resulting array to output vector
	std::vector<int> v_out (a_out, a_out + N_out);

	#if DEBUG
	printf("\tOutput = [ ");
	for (int i=0; i<v_out.size(); ++i) {
		if (i == v_out.size()-1) {
			printf("%d ]\n", v_out[i]);
		} else {
			printf("%d, ", v_out[i]);
		}
	}
	#endif


	// Free GPU memory
	cudaFree(d_in);
	cudaFree(d_out);

	// Free host memory
	free(a_out);

	// Save output to file
	write_output(OUTPUT_FILE_Q2B, v_out);
}

/** Q2 c) Compute a counter array in global memory of GPU
 *
 * Create an array of size 10 that uses B to compute C which keeps count of the
 * entries in each of the ranges:[0,99], [0,199], [0,299],. . . , [0, 999]. Note
 * that the ranges are different from the part (a). For this part of the
 * problem, you must not use array A.
 *
 * \param	v_in		Input array as a vector
 * \param	dev_props	CUDA device properties
 */
void q2c (const std::vector<int> &v_in, cudaDeviceProp *dev_props) {
	#if DEBUG
	printf("\tTransfering input array to GPU memory...\n");
	#endif

	// Declare GPU memory pointers
	int *d_in;

	// Allocate GPU memory
	int N = v_in.size();					// Problem size (input array size)
	int d_in_size = N * sizeof(int);		// Input array size in bytes

	#if DEBUG
	printf("\tN (input array size): %d\n", N);
	#endif

	/*
	if (N > ((int)((*dev_props).maxThreadsPerBlock) * (int)((*dev_props).maxThreadsPerBlock))) {
		fprintf(stderr, "ERROR:q1a: problem size (input array size) is too large\n");
		exit(EXIT_FATAL);
	}
	*/

	cudaMalloc((void **) &d_in, d_in_size);

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
	
	printf("\tCounting entries using result from Q2 a...\n");
	#endif

	// Calculate the number of blocks and threads to use
	int threads_per_block = (int)((*dev_props).maxThreadsPerBlock); // Max number of threads per block
	int blocks_per_grid = (N + (threads_per_block - 1)) / threads_per_block;

	#if DEBUG
	printf("\tThreads per block: %d\n", threads_per_block);
	printf("\tBlocks per grid: %d\n", blocks_per_grid);
	printf("\tRunning kernel...\n");
	cudaEventRecord(start, 0);
	#endif

	// Launch the kernel to find min
	parallelScanSumKernel<<<blocks_per_grid, threads_per_block>>>
		(d_in, N);
	
	// Make sure all the blocks finish executing
	cudaDeviceSynchronize();
	cudaDeviceSynchronize();
	
	#if DEBUG
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// Calculate elapsed time, and print it
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("\tAverage time elapsed: %f\n", elapsedTime);
	#endif

	// Copy back the result from GPU
	int *a_out;
	a_out = (int*) malloc(d_in_size);
	cudaMemcpy(a_out, d_in, d_in_size, cudaMemcpyDeviceToHost);

	#if DEBUG >= 2
	printf("\ta_out = [ ");
	for (int i=0; i<N; ++i) {
		if (i == N-1) {
			printf("%d ]\n", a_out[i]);
		} else {
			printf("%d, ", a_out[i]);
		}
	}
	#endif
	
	// Copy resulting array to output vector
	std::vector<int> v_out (a_out, a_out + N);

	#if DEBUG
	printf("\tOutput = [ ");
	for (int i=0; i<v_out.size(); ++i) {
		if (i == v_out.size()-1) {
			printf("%d ]\n", v_out[i]);
		} else {
			printf("%d, ", v_out[i]);
		}
	}
	#endif


	// Free GPU memory
	cudaFree(d_in);

	// Free host memory
	free(a_out);

	// Save output to file
	write_output(OUTPUT_FILE_Q2C, v_out);
}

/** Main
 *
 * Set up CUDA device, read input file, and run Q2a, Q2b and Q2c.
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
	std::vector<int> v_out;
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
	std::printf("Running Q2 a...\n");
	#endif

	// Problem q2 a
	v_out = q2a(v_in, &dev_props);

	/*
	#if DEBUG
	std::printf("Reseting device...\n");
	#endif

	cudaDeviceReset();
	*/

	#if DEBUG
	std::printf("Running Q2 b...\n");
	#endif

	// Problem q2 b
	q2b(v_in, &dev_props);

	/*
	#if DEBUG
	std::printf("Reseting device...\n");
	#endif

	cudaDeviceReset();
	*/

	#if DEBUG
	std::printf("Running Q2 c...\n");
	#endif

	// Problem q2 c
	q2c(v_out, &dev_props);

	/*
	#if DEBUG
	std::printf("Reseting device...\n");
	#endif

	cudaDeviceReset();
	*/

	#if DEBUG
	std::printf("Done\n");
	#endif

	return 0;
}
