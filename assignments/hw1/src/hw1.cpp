#include <iostream>
#include <omp.h>
#include "hw1.h"

double euclidean_length(std::vector<double> vector) {
  
  if(vector.size()==0){
    return 0;
  }
  int i;
  double sum=0;
  double result=0;

  #pragma opm parallel for shared(sum, vector) reduciton(+: sum) 
  std:: cout<< "Number of therad: "<<omp_get_thread_num()<<"\n";
  for(auto i=0;i<vector.size();i++){
    sum += vector[i]*vector[i];
  }
  result=sqrt(sum);
  return result;
}

std::vector<int64_t> discard_duplicates(std::vector<int64_t> sorted_vector) {
  // Your code goes here
  return std::vector<int64_t>();
}
