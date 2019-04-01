#include "mnblas.h"
#include "complexe.h"

float  mnblas_sasum(const int N, const float *X, const int incX) {
  register unsigned i = 0;
  float r = 0.0;
  for (; i < N; i += incX) {
    if (X[i] < 0) r = r - X[i];
    else r = r + X[i];
  }
  return r;
}

double mnblas_dasum(const int N, const double *X, const int incX) {
  register unsigned i = 0;
  double r = 0.0;
  for (; i < N; i += incX) {
    if (X[i] < 0) r = r - X[i];
    else r = r + X[i];
  }
  return r;
}

float  mnblas_scasum(const int N, const void *X, const int incX) {
  register unsigned i = 0;
  complexe_float_t * x = (complexe_float_t*)X;
  float r = 0.0;
  for (; i < N; i += incX) {
    if (x[i].real < 0) r = r - x[i].real;
    else r = r + x[i].real;
    if (x[i].imaginary < 0) r = r - x[i].imaginary;
    else r = r + x[i].imaginary;
  }
  return r;
}

double mnblas_dzasum(const int N, const void *X, const int incX) {
  register unsigned i = 0;
  complexe_double_t * x = (complexe_double_t*)X;  
  double r = 0.0;
  for (; i < N; i += incX) {
    if (x[i].real < 0) r = r - x[i].real;
    else r = r + x[i].real;
    if (x[i].imaginary < 0) r = r - x[i].imaginary;
    else r = r + x[i].imaginary;
  }
  return r;
}
