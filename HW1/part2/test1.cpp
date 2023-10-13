#include <iostream>
#include "test.h"
#include "fasttime.h"

void test1(float* __restrict a, float* __restrict b, float* __restrict c, int N) {
  __builtin_assume(N == 1024);
  a = (float *)__builtin_assume_aligned(a, 16);
  b = (float *)__builtin_assume_aligned(b, 16);
  c = (float *)__builtin_assume_aligned(c, 16);
 
  double elapsedf = 0;
  for(int t=0;t<10;t++)
  {
  fasttime_t time1 = gettime();
  for (int i=0; i<I; i++) {
    for (int j=0; j<N; j++) {
      c[j] = a[j] + b[j];
    }
  }
  fasttime_t time2 = gettime();
  elapsedf += tdiff(time1, time2);
  }
  std::cout << "Elapsed execution time of the loop in test1() 10 times:\n" 
    << elapsedf << "total sec (N: " << N << ", I: " << I << ")\n"
	<< elapsedf / 10 << "avg sec (N: " << N << ", I: " << I << ")\n";
}
