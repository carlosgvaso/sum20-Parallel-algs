#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <cmath>
#include "hw1.h"

#define DEBUG 0 // Enable (1) / disable (0) debugging output

/*
 * Allows us to easily print a vector in the format [#, #, #]
 */
template<typename T>
std::ostream& operator<<(std::ostream& s, const std::vector<T>& v) 
{
    s.put('[');
    char comma[3] = {'\0', ' ', '\0'};
    for (const auto& e : v) {
        s << comma << e;
        comma[0] = ',';
    }
    return s << ']';
}

double euclidean_length(std::vector<double> vector) {
  // Your code goes here.
  int n = vector.size();

  if(n <= 0){
    return 0;
  }

  int i;
  int chunk = 1;
  double sum = 0.0;
  double result = 0.0;

#pragma omp parallel for default(shared) private(i) schedule(static, chunk) \
reduction(+:sum)
  for(i=0; i<n; i++){
    sum += vector[i]*vector[i];

    // Debugging
#if DEBUG == 1
    std::cout << "Number of threads: " << omp_get_thread_num() << std::endl;
#endif
  }

  result=sqrt(sum);

  return result;
}

std::vector<int64_t> discard_duplicates(std::vector<int64_t> sorted_vector) {
  // Your code goes here

  // Debugging
#if DEBUG == 1
  std::cout << "A: " << sorted_vector << std::endl;
#endif

  // Get size of input vector also known as vector A
  int64_t n = sorted_vector.size();

  // If the vector has size of 1 or less, it can't have any repeated values
  if(n <= 1){
    return sorted_vector;
  }

  // Declare auxiliary vectors C and D
  std::vector<int64_t> C(n), D(n);
  int64_t i, i_step, i_step_l, h, h_max, l;

  /*
   * Create an array C such that C[i] = 1, if A[i] != A[i−1],
   * and 0 otherwise (A is the input vector sorted_vector)
   */
#pragma omp parallel for default(shared) private(i)
  for (i=0; i<n; i++) {
    if (i == 0) { // First entry is always 1
      C[i] = 1;
    } else if (sorted_vector[i] != sorted_vector[i-1]) {
      C[i] = 1;
    } else {
      C[i] = 0;
    }
  }

  // Debugging
#if DEBUG == 1
  std::cout << "C: " << C << std::endl;
#endif

  // Create array D as the prefix sum of C
  // Set up auxiliary vector D
#pragma omp parallel for default(shared) private(i)
  for (i=0; i<n; i++) {
    D[i] = C[i];
  }

  // Debugging
#if DEBUG == 1
  std::cout << "D1: " << D << std::endl;
#endif

  // Upward sweep (reduce operation)
  h_max = ((int64_t)log2((double)n)) - 1;
  for (h=0; h<=h_max; h++) {
    i_step = (int64_t)pow(2.0, (double)(h+1));
    i_step_l = (int64_t)pow(2.0, (double)h);
#pragma omp parallel for default(shared) private(i)
    for (i=0; i<n; i+=i_step) {
      D[i + i_step - 1] = D[i + i_step_l - 1] + D[i + i_step - 1];
    }
  }

  // Debugging
#if DEBUG == 1
  std::cout << "D2: " << D << std::endl;
#endif

  // Downward sweep
  D[n-1] = 0;
  for (h=h_max; h>=0; h--) {
    i_step = (int64_t)pow(2.0, (double)(h+1));
    i_step_l = (int64_t)pow(2.0, (double)h);
#pragma omp parallel for default(shared) private(i, l)
    for (i=0; i<n; i+=i_step) {
      l = D[i + i_step_l - 1];
      D[i + i_step_l - 1] = D[i + i_step - 1];
      D[i + i_step - 1] = l + D[i + i_step - 1];
    }
  }

  // Debugging
#if DEBUG == 1
  std::cout << "D3: " << D << std::endl;
#endif

  /*
   * Format the D vector so that the values getting written concurrently are the
   * the same values (common concurrent write), and any collissions caused by
   * duplicated values are solved that way. This can be done by transforming the
   * exclusive scan to inclusive scan, and subtracting all entries by 1 so that
   * the indexes start at 0.
   */
#pragma omp parallel for default(shared) private(i)
  for (i=0; i<n; i++) {
    D[i] = D[i] + C[i] - 1;
  }

  // Debugging
#if DEBUG == 1
  std::cout << "D4: " << D << std::endl;
#endif

  /*
   * Calculate the solution array B from D, such that if D[i] = k, then the
   * number of distinct elements in A which are smaller than or equal A[i] is k.
   * Therefore, A[i] should go in the kth position in B. Notice, B is of size
   * D[n-1] + 1 (last entry of vector D plus the 1 we removed earlier), since
   * we used an inclusive scan.
   */
  std::vector<int64_t> B(D[n-1] + 1);

#pragma omp parallel for default(shared) private(i)
  for (i=0; i<n; i++) {
    if (i == 0) { // First entry is always added
      B[D[i]] = sorted_vector[i];
    } else if (C[i] == 1) { //(sorted_vector[i] != sorted_vector[i-1]) {
      B[D[i]] = sorted_vector[i];
    }
  }

  return B;
}
