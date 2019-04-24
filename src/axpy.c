#include "mnblas.h"
#include "complexe.h"
#include <smmintrin.h>
#include <x86intrin.h>

void mnblas_saxpy(const int N, const float alpha, const float *X,
                 const int incX, float *Y, const int incY)
{
  __m128 v1 ;
  register unsigned int i ;
  float axpy [4] __attribute__ ((aligned (16))) ;

  #pragma omp parallel for private(v1, axpy)
    for (i = 0; i < N; i = i + 4)
    {
      v1 = _mm_load_ps (X+i) ;

      _mm_store_ps (axpy, v1) ;

      Y[i] += axpy[0]*alpha ;
      Y[i+1] += axpy[1]*alpha ;
      Y[i+2] += axpy[2]*alpha ;
      Y[i+3] += axpy[3]*alpha ;
    }

    return ;
}

void mnblas_daxpy(const int N, const double alpha, const double *X,
                 const int incX, double *Y, const int incY)
{
  __m128d v1 ;
  register unsigned int i ;
  double axpy [2] __attribute__ ((aligned (16))) ;

  #pragma omp parallel for
    for (i = 0; i < N; i = i + 2)
    {
      v1 = _mm_load_pd (X+i) ;

      _mm_store_pd (axpy, v1) ;

      Y[i] += axpy[0]*alpha ;
      Y[i+1] += axpy[1]*alpha ;
    }

    return ;

}

void mnblas_caxpy(const int N, const void *alpha, const void *X,
                 const int incX, void *Y, const int incY)
{
  float axpy_r = 0 ;
  float axpy_i = 0 ;
  complexe_float_t* x = (complexe_float_t*)X;
  complexe_float_t* y = (complexe_float_t*)Y;
  complexe_float_t* a = (complexe_float_t*)alpha;
  
  #pragma omp parallel for private(axpy_r, axpy_i)
  for (int i = 0; i < N ; i += incX)
    {
      axpy_r = a->real * x[i].real - a->imaginary * x[i].imaginary + y[i].real;
      axpy_i = x[i].real * a->imaginary + x[i].imaginary * a->real + y[i].imaginary;
      y[i].real = axpy_r;
      y[i].imaginary = axpy_i;
    }
    
}

void mnblas_zaxpy(const int N, const void *alpha, const void *X,
                 const int incX, void *Y, const int incY)
{
  double axpy_r = 0 ;
  double axpy_i = 0 ;
  complexe_double_t* x = (complexe_double_t*)X;
  complexe_double_t* y = (complexe_double_t*)Y;
  complexe_double_t* a = (complexe_double_t*)alpha;
  
  #pragma omp parallel for private(axpy_r, axpy_i)
  for (int i = 0; i < N ; i += incX)
    {
      axpy_r = a->real * x[i].real - a->imaginary * x[i].imaginary + y[i].real;
      axpy_i = x[i].real * a->imaginary + x[i].imaginary * a->real + y[i].imaginary;
      y[i].real = axpy_r;
      y[i].imaginary = axpy_i;
    }
}
