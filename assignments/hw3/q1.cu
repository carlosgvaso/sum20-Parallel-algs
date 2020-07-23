/** Homework 3 question 1 code
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
#define OUTPUT_FILE_Q1A "q1a.txt"
#define OUTPUT_FILE_Q1B "q1b.txt"


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
	if (!fin.is_open()) throw std::runtime_error("Q1:read_input: Could not open file");

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


/** Main
 *
 * \param	argc	Number of command-line arguments
 * \param	argv	Array of command-line arguments
 * \return	Program return code
 */
int main (int argc, char **argv) {
	/* Test ===================================================================
	 *
	 * Read input and write to output 1a file
	 */
	std::vector<int> arr_in;

	arr_in = read_input(INPUT_FILE);
	write_output(OUTPUT_FILE_Q1A, arr_in);
	// Test end ===============================================================

	return 0;
}

