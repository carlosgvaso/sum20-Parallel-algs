#include <iostream>
#include <vector>
#include "hw1.h"

#define DEBUG 0 // Enable (1) / disable (0) debugging output

int main() {
  // Example execution for problem 5
  std::vector<double> vector = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::cout << "Euclidean distance: " << euclidean_length(vector) << std::endl; // This should print 19.62...

  // Example execution for problem 6
#if DEBUG == 0
  std::vector<int64_t> sorted_with_dups = {1, 2, 2, 4, 5, 5, 6, 6};
  std::vector<int64_t> result = discard_duplicates(sorted_with_dups);

  //  This should print {1, 2, 4, 5, 6}
  std::cout << "Sorted vector: ";
  for (auto i = result.begin(); i != result.end(); ++i)
    std::cout << *i << ' ';
  std::cout << std::endl;

#else
  std::vector<std::vector<int64_t>> vectors_with_dups {
    {},
    {5},
    {2, 5},
    {5, 5},
    {1, 1, 2, 3},
    {1, 2, 3, 3},
    {1, 2, 2, 3},
    {1, 2, 3, 4},
    {1, 2, 2, 4, 5, 5, 6, 6},
    {1, 1, 1, 2, 2, 4, 5, 5, 6, 6, 6, 6, 7, 9, 10, 16}
  };

  std::vector<std::vector<int64_t>> vectors_without_dups {
    {},
    {5},
    {2, 5},
    {5},
    {1, 2, 3},
    {1, 2, 3},
    {1, 2, 3},
    {1, 2, 3, 4},
    {1, 2, 4, 5, 6},
    {1, 2, 4, 5, 6, 7, 9, 10, 16}
  };

  for (int j=0; j<vectors_with_dups.size(); j++) {
    for (int k=0; k<5; k++) {
      std::vector<int64_t> result = discard_duplicates(vectors_with_dups[j]);

      //  This will print the result
      std::cout << "Sorted vector: ";
      for (auto i = result.begin(); i != result.end(); ++i)
        std::cout << *i << ' ';
      std::cout << std::endl;

      if (result != vectors_without_dups[j]) {
        std::cout << "Wrong answer!" << std::endl;
      } else {
        std::cout << "Right answer!" << std::endl;
      }
    }
  }
#endif

  return 0;
}
