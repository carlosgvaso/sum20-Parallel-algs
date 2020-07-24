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
#define DEBUG 1	//! Enable debug messages

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
void write_output (std::string filename, std::vector<int> v_out) {
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
__global__ void arrayMinKernel(int *d_out, int *d_in)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    //int tid  = threadIdx.x;

	/* TODO: Implement Hillis-Steele parallel scan min
    // do reduction in global mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            d_in[myId] += d_in[myId + s];
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = d_in[myId];
	}
	*/
	if (myId == 0)
    {
        d_out[0] = d_in[0];
    }
}

/** Q1 a) Compute minA, the minimum value in the input array
 *
 * This function uses the Hillis-Steele version of parallel scan to find the
 * minimum value in the input array. Then, it outputs the result to the
 * OUTPUT_FILE_Q1A output file.
 *
 * \param v_in	Input array as a vector
 */
void q1a (std::vector<int> v_in) {
	#if DEBUG
	printf("\tTransfering input array to GPU memory...\n");
	#endif

	// Declare GPU memory pointers
	//int *d_in, *d_intermediate, *d_out;
	int *d_in, *d_out;

    // Allocate GPU memory
    cudaMalloc((void **) &d_in, v_in.size());
    //cudaMalloc((void **) &d_intermediate, ARRAY_BYTES); // overallocated
    cudaMalloc((void **) &d_out, sizeof(int));

	/* Transfer the input array to the GPU
	 * Since the elements of a vector are stored contiguously in memory, we can
	 * pass a pointer to the first element of the vector, and that will act as
	 * if we passed a C array.
	 */
    cudaMemcpy(d_in, &v_in[0], v_in.size(), cudaMemcpyHostToDevice);

	/* TODO: Calculate the number of blocks and threads to use
	 *
	 * This, assumes that size is not greater than maxThreadsPerBlock^2, and
	 * that size is a multiple of maxThreadsPerBlock. However, this is not a
	 * valid assumption.
	 */
	const int maxThreadsPerBlock = 512;
	int threads = maxThreadsPerBlock;
    int blocks = v_in.size() / maxThreadsPerBlock;

	#if DEBUG
	// Set up a timer to measure the elapsed time to find the min
	cudaEvent_t start, stop;
    cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	printf("\tFinding minimum entry in the array...\n");
	cudaEventRecord(start, 0);
	#endif

	// Launch the kernel to find min
	//arrayMin(d_out, d_in, v_in.size());
	arrayMinKernel<<<blocks, threads>>>(d_out, d_in);
	
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
	cudaMemcpy(&a_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

	#if DEBUG
	printf("\tMin: %d\n", a_out);
	#endif
	
	// Copy resulting array to output vector
	std::vector<int> v_out;
	v_out.push_back(a_out);

	// Free GPU memory
	cudaFree(d_in);
	//cudaFree(d_intermediate)
	cudaFree(d_out);

	write_output(OUTPUT_FILE_Q1A, v_out);
}

/** Q1 b) Compute an array B such that B[i] is the last digit of A[i] for all i
 *
 * \param v_in	Input array as a vector
 */
 void q1b (std::vector<int> v_in) {
	std::vector<int> v_out;

	// TODO: Implement
	v_out = v_in;

	write_output(OUTPUT_FILE_Q1B, v_out);
}


/** Main
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
	int deviceCount;
	int dev = 0;
	cudaDeviceProp devProps;

	#if DEBUG
	printf("Detecting CUDA devices...\n");
	#endif

	// Check there are CUDA devices available
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		fprintf(stderr, "ERROR:main: no CUDA devices found\n");
		exit(EXIT_FATAL);
	}

	// Use device 0
	cudaSetDevice(dev);

	#if DEBUG
	if (cudaGetDeviceProperties(&devProps, dev) == 0) {
		printf("\tDevice ID: %d\n"
			   "\tName: %s\n"
			   "\tGlobal mem: %d B\n"
			   "\tCompute: v%d.%d\n"
			   "\tClock: %d kHz\n",
			   dev,
			   devProps.name,
			   (int)devProps.totalGlobalMem,
			   (int)devProps.major,
			   (int)devProps.minor,
			   (int)devProps.clockRate);
	} else {
		fprintf(stderr, "ERROR:main: could not find CUDA device information\n");
	}
	#endif

	#if DEBUG
	std::printf("Reading input array...\n");
	#endif

	// Read input array
	v_in = read_input(INPUT_FILE);

	#if DEBUG
	std::printf("Running Q1 a...\n");
	#endif

	// Problem q1 a
	q1a(v_in);

	#if DEBUG
	std::printf("Running Q1 b...\n");
	#endif

	// Problem q1 b
	q1b(v_in);

	#if DEBUG
	std::printf("Done\n");
	#endif

	return 0;
}

