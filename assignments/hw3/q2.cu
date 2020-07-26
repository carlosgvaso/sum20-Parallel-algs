/** Homework 3 question 2 code
 *
 * \file q1.cu
 * \author Jose Carlos Martinez Garcia-Vaso <carlosgvaso@utexas.edu>
 * \author Utkarsh Vardan <uvardan@utexas.edu>
 */


 #include <cstdio>		// standard I/O
 #include <string>		// strings
 #include <fstream>		// streams
 #include <vector>		// std::vector
 #include <stdexcept>	// std::runtime_error
 #include <sstream>		// std::stringstream
 
 
 // Globals
 #define INPUT_FILE "inp.txt"
 #define OUTPUT_FILE_Q2A "q2a.txt"
 #define OUTPUT_FILE_Q1B "q2b.txt"
 
 
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
 
 
 __global__ void unique_idx_calc_threadIdx(int* input, int* result)
 {
	 //printf("result : %d \n",result[blockIdx.x]);
	 int tid=threadIdx.x;
	 int offset=blockIdx.x * blockDim.x;
	 int gid=tid + offset;
	 if(input[tid] != NULL){
	 result[blockIdx.x]=1;
	 }
	 printf("blockIdx.x: %d, threadIdx: %d, gid: %d, value : %d, b: %d \n",blockIdx.x, tid, gid, input[tid], result[blockIdx.x]);
	 
 }
 
 /** Main
  *
  * \param	argc	Number of command-line arguments
  * \param	argv	Array of command-line arguments
  * \return	Program return code
  */
 int main (int argc, char **argv) {
	 /* Test ===================================================================
	  *
	  * Read input and write to output to file
	  */
	 std::vector<int> arr_in;
	 int* arr_input;
	 arr_in = read_input(INPUT_FILE);
	 arr_input=arr_in.data();
	 int size_arr;
	 int size_B;
	 size_B=10;
	 int B[size_B];
	 int* d_B;
	 for (int i=0;i<size_B;i++){
	 B[i]=0;
	 }
	 size_arr=arr_in.size();
	 printf("Size of array %d \n", arr_in.size());
		 int *d_arr_in;
	 cudaMalloc((void**)&d_arr_in,size_arr);
	 cudaMemcpy(d_arr_in, arr_input, size_arr, cudaMemcpyHostToDevice);
	 cudaMalloc((void**)&d_B,size_B);
	 cudaMemcpy(d_B,B, size_B,cudaMemcpyHostToDevice);
	 dim3 block(100);
	 dim3 grid (10);
	 unique_idx_calc_threadIdx<<<grid,block>>>(d_arr_in,d_B);
	 cudaDeviceSynchronize();
	 
	 cudaMemcpy(B,d_B,size_B,cudaMemcpyDeviceToHost);
	 
	 for(int i=0;i<size_B;i++){
		 printf("%d \n", B[i]);
	 }
 
	 cudaDeviceReset();
 
	 //write_output(OUTPUT_FILE_Q2A, d_B);
	 // Test end ===============================================================
 
	 return 0;
 }
 
 
 