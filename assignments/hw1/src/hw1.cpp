#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <cmath>
#include "hw1.h"

double euclidean_length(std::vector<double> vector) {
  // Your code goes here.
  int n = vector.size();

  if(n <= 0){
    return 0;
  }

  int i;
  int chunk = n/2;
  double sum = 0.0;
  double result = 0.0;


#pragma opm parallel for default(shared) private(i) schedule(static,chunk) reduction(+:sum)
  for(i=0; i<n; i++){
    sum += vector[i]*vector[i];

    //std::cout << "Number of threads: " << omp_get_thread_num() << std::endl;
  }

  result=sqrt(sum);

  return result;
}
std::vector<int64_t> discard_duplicates(std::vector<int64_t> sorted_vector) {
  // Your code goes here
  int n = sorted_vector.size();

  if(n <= 1){
    return sorted_vector;
  }

  // Declare vector to return with no duplicates
  std::vector<int64_t> no_dup_vector;
/* WIP
  // Get size of input vector
  int64_t n = sorted_vector.size();

  for (int h=1; h<=ceil(log2((double)n)); h++) {
#pragma omp parallel for shared(sorted_vector, no_dup_vector)
    for (int i=1; i<=ceil(n/h); i+=2) {
      //
    }
  }
*/
  return no_dup_vector;
}
