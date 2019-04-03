#include "mnblas.h"
#include "complexe.h"

float mncblas_sdot(const int N, const float *X, const int incX,
                 const float *Y, const int incY)
{
  register float dot = 0.0 ;
  #pragma omp parallel for reduction(+:dot)
  for (int i = 0; i < N; i += incX)
  {
    dot = dot + X [i] * Y [i] ;
  }
  return dot ;
}

double mncblas_ddot(const int N, const double *X, const int incX,
                 const double *Y, const int incY)
{
  register double dot = 0.0;
  #pragma omp parallel for reduction(+:dot)
  for (int i = 0; i < N ; i += incX) {
    dot = dot + X [i] * Y [i] ;
  }
  
  return dot;
}

void   mncblas_cdotu_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotu)
{
  float dot_r = 0;
  float dot_i = 0;
  register complexe_float_t* dot = (complexe_float_t*)dotu;
  complexe_float_t* x = (complexe_float_t*)X;
  complexe_float_t* y = (complexe_float_t*)Y;

  #pragma omp parallel for reduction(+:dot_r) reduction(+:dot_i)
  for (int i = 0; i < N; i += incX) {
    dot_r += x[i].real * y[i].real - x[i].imaginary * y[i].imaginary;
    dot_i += x[i].real * y[i].imaginary + x[i].imaginary * y[i].real;
  }
  
  dot->real = dot_r;
  dot->imaginary = dot_i;
}

void   mncblas_cdotc_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotc)
{
  float dot_r = 0;
  float dot_i = 0;
  register complexe_float_t* dot = dotc;
  complexe_float_t* x = (complexe_float_t*)X;
  complexe_float_t* y = (complexe_float_t*)Y;
  
  #pragma omp parallel for reduction(+:dot_r) reduction(+:dot_i)
  for (int i = 0; i < N; i += incX) {
    dot_r += x[i].real * y[i].real - (-x[i].imaginary) * y[i].imaginary;
    dot_i += x[i].real * y[i].imaginary + (-x[i].imaginary) * y[i].real;
  }

  dot->real = dot_r;
  dot->imaginary = dot_i;
}

void   mncblas_zdotu_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotu)
{
  double dot_r = 0;
  double dot_i = 0;
  register complexe_double_t* dot = dotu;
  complexe_double_t* x = (complexe_double_t*)X;
  complexe_double_t* y = (complexe_double_t*)Y;
  
  #pragma omp parallel for reduction(+:dot_r) reduction(+:dot_i)
  for (int i = 0; i < N; i += incX) {
    dot_r += x[i].real * y[i].real - x[i].imaginary * y[i].imaginary;
    dot_i += x[i].real * y[i].imaginary + x[i].imaginary * y[i].real;
  }

  dot->real = dot_r;
  dot->imaginary = dot_i;
}

void   mncblas_zdotc_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotc)
{
  double dot_r = 0;
  double dot_i = 0;
  register complexe_double_t* dot = dotc;
  complexe_double_t* x = (complexe_double_t*)X;
  complexe_double_t* y = (complexe_double_t*)Y;
  
  #pragma omp parallel for reduction(+:dot_r) reduction(+:dot_i)
  for (int i = 0; i < N; i += incX) {
    dot_r += x[i].real * y[i].real - (-x[i].imaginary) * y[i].imaginary;
    dot_i += x[i].real * y[i].imaginary + (-x[i].imaginary) * y[i].real;
  }

  dot->real = dot_r;
  dot->imaginary = dot_i;
}
