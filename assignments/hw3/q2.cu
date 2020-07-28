/** Homework 3 question 2 code
 *
 * \file q2.cu
 * \author Utkarsh Vardan <uvardan@utexas.edu>
 * \author Jose Carlos Martinez Garcia-Vaso <carlosgvaso@utexas.edu>
 */


#include <cstdio>		// standard I/O
#include <string>		// strings
#include <fstream>		// streams
#include <vector>		// std::vector
#include <stdexcept>	// std::runtime_error
#include <sstream>		// std::stringstream


// Globals
#define DEBUG 0	//! Enable debug messages (0: no messages, 1: some messages, 2: all messages)

#define INPUT_FILE "inp.txt"
#define OUTPUT_FILE_Q2A "q2a.txt"
#define OUTPUT_FILE_Q1B "q2b.txt"

#define EXIT_OK 0		//! Exit code success
#define EXIT_FATAL 1	//! Exit code unrecoverable error

#define SHARED_ARRAY_SIZE 10


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
	if (!fin.is_open()) throw std::runtime_error("Q2:read_input: Could not open file");

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

// 2.a 
__global__ void unique_idx_calc_threadIdx(int * input, int * d_B, int size)
{
	//printf("result : %d \n",result[blockIdx.x]);
	int tid=threadIdx.x;
	int blockid=blockIdx.x;
	int offset= blockid * blockDim.x;
	int gid=tid + offset;
	if(gid<size){
		if(input[gid]>=0 && input[gid]<100){
			atomicAdd(&d_B[0],1);
		}
		if(input[gid]>=100 && input[gid]<200){
			atomicAdd(&d_B[1],1);
		}
		if(input[gid]>=200 && input[gid]<300){
			atomicAdd(&d_B[2],1);
		}
		if(input[gid]>=300 && input[gid]<400){
			atomicAdd(&d_B[3],1);
		}
		if(input[gid]>=400 && input[gid]<500){
			atomicAdd(&d_B[4],1);
		}
		if(input[gid]>=500 && input[gid]<600){
			atomicAdd(&d_B[5],1);
		}
		if(input[gid]>=600 && input[gid]<700){
			atomicAdd(&d_B[6],1);
		}
		if(input[gid]>=700 && input[gid]<800){
			atomicAdd(&d_B[7],1);
		}
		if(input[gid]>=800 && input[gid]<900){
			atomicAdd(&d_B[8],1);
		}
		if(input[gid]>=900 && input[gid]<1000){
			atomicAdd(&d_B[9],1);
		}
	}
	__syncthreads(); 
}

// 2.b
__global__ void sharedCount(int * input, int * d_B, int size)
{
	//printf("result : %d \n",result[blockIdx.x]);
	int tid=threadIdx.x;
	int blockid=blockIdx.x;
	int offset= blockid * blockDim.x;
	int gid=tid + offset;
	extern __shared__ int B_Array[];
	B_Array[tid]=input[gid];
	if(gid<size){
		if(B_Array[tid]>=0 && B_Array[tid]<100){
			atomicAdd(&d_B[0],1);
		}
		if(B_Array[tid]>=100 && B_Array[tid]<200){
			atomicAdd(&d_B[1],1);
		}
		if(B_Array[tid]>=200 && B_Array[tid]<300){
			atomicAdd(&d_B[2],1);
		}
		if(B_Array[tid]>=300 && B_Array[tid]<400){
			atomicAdd(&d_B[3],1);
		}
		if(B_Array[tid]>=400 && B_Array[tid]<500){
			atomicAdd(&d_B[4],1);
		}
		if(B_Array[tid]>=500 && B_Array[tid]<600){
			atomicAdd(&d_B[5],1);
		}
		if(B_Array[tid]>=600 && B_Array[tid]<700){
			atomicAdd(&d_B[6],1);
		}
		if(B_Array[tid]>=700 && B_Array[tid]<800){
			atomicAdd(&d_B[7],1);
		}
		if(B_Array[tid]>=800 && B_Array[tid]<900){
			atomicAdd(&d_B[8],1);
		}
		if(B_Array[tid]>=900 && B_Array[tid]<1000){
			atomicAdd(&d_B[9],1);
		}
	}
	__syncthreads();
}

//2.c
__global__ void scan(int *d_B, int size)
{
	//printf("result : %d \n",result[blockIdx.x]);
	int tid=threadIdx.x;
	int blockid=blockIdx.x;
	int offset= blockid * blockDim.x;
	int gid=tid + offset;
	if(gid<size){
		for(int offset=1;offset<=tid;offset *= 2){
			d_B[gid]+=d_B[gid-offset];
			__syncthreads();
		}
	}
	__syncthreads();
}

/** Main
 *
 * \param	argc	Number of command-line arguments
 * \param	argv	Array of command-line arguments
 * \return	Program return code
 */
int main (int argc, char **argv) {
	std::vector<int> arr_in;
	arr_in = read_input(INPUT_FILE);
	int* arr_input;
	arr_input=arr_in.data();
	int size_arr=arr_in.size();
	int size_B=10;
	int B[size_B];
	int C[size_B];
	for(int i=0;i<size_B;i++){
		B[i]=0;
		C[i]=0;
	}
	//Global variables
	int* d_B;
	int* d_C;
	int *d_arr_in;
	int size_arr_byte=sizeof(int) * size_arr;
	int size_B_byte=sizeof(int)*size_B;
	cudaMalloc((void**)&d_arr_in,size_arr_byte);
	cudaMemcpy(d_arr_in, arr_input, size_arr_byte, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_B,size_B_byte);
	cudaMemcpy(d_B,B, size_B_byte,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_C,size_B_byte);
	cudaMemcpy(d_C,C, size_B_byte,cudaMemcpyHostToDevice);
	dim3 block(32);
	dim3 grid (((size_arr/32)+1));
	//unique_idx_calc_threadIdx<<<grid,block>>>(d_arr_in,d_B, size_arr);
	sharedCount<<<grid,block, sizeof(int)*SHARED_ARRAY_SIZE>>>(d_arr_in,d_B,size_arr);
	//cudaMemcpy(&B,d_B,size_B_byte,cudaMemcpyDeviceToHost);
	scan<<<1,block>>>(d_B,size_B);
	cudaMemcpy(&C,d_B,size_B_byte,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	for(int i=0;i<size_B;i++){
		printf("%d \n", C[i]);
	}

	cudaDeviceReset();

	return 0;
}
