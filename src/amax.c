#include "mnblas.h"
#include "complexe.h"
#include <math.h>
CBLAS_INDEX mnblas_isamax(const int N, const float  *X, const int incX){
  if(incX < 0 || N < 0) return 0;
  int max = abs(X[0]);
  CBLAS_INDEX indMax = 0;
  for(int i =0; i < N ;i = i +incX){
    if(max<abs(X[i])){
      max = abs(X[i]);
      indMax = i;
    }
  }
  return indMax;
}
CBLAS_INDEX mnblas_idamax(const int N, const double *X, const int incX){
  if(incX < 0 || N < 0) return 0;
  int max = abs(X[0]);
  CBLAS_INDEX indMax = 0;
  for(int i =0; i < N ;i = i +incX){
    if(max<abs(X[i])){
      max = abs(X[i]);
      indMax = i;
    }
  }
  return indMax;
}
CBLAS_INDEX mnblas_icamax(const int N, const void   *X, const int incX){
  if(incX < 0 || N < 0) return 0;
  complexe_float_t * x = (complexe_float_t*)X;
  int max = abs(x[0].real) + abs(x[0].imaginary);
  CBLAS_INDEX indMax = 0;
  for(size_t i =1; i < N ;i = i +incX){
    if(max<abs(x[i].real)+abs(x[i].imaginary)){
      max = abs(x[i].real)+abs(x[i].imaginary);
      indMax = i;
    }
  }
  return indMax;
}

CBLAS_INDEX mnblas_izamax(const int N, const void   *X, const int incX){
  if(incX < 0 || N < 0) return 0;
  complexe_double_t * x = (complexe_double_t*)X;
  int max = abs(x[0].real) + abs(x[0].imaginary);
  CBLAS_INDEX indMax = 0;
  for(int i =1; i < N ;i = i +incX){
    if(max<abs(x[i].real)+abs(x[i].imaginary)){
      max = abs(x[i].real)+abs(x[i].imaginary);
      indMax = i;
    }
  }
  return indMax;
}
