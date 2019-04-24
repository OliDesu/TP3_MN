#include "../include/mnblas.h"
#include "../include/complexe.h"
#include <smmintrin.h>
#include <x86intrin.h>
void mncblas_scopy(const int N, const float *X, const int incX,
                 float *Y, const int incY)
{
  __m128 v1 ;
  register unsigned int i ;
  float copy [4] __attribute__ ((aligned (16))) ;
  #pragma omp parallel for private(v1, copy)
  for (i = 0; i < N; i += 4)
    {
      v1 = _mm_load_ps (X+i) ;

      _mm_store_ps (copy, v1) ;
      Y [i] = copy [0] ;
      Y [i+1] = copy [1] ;
      Y [i+2] = copy [2] ;
      Y [i+3] = copy [3] ;
    }

  return ;
}

void mncblas_dcopy(const int N, const double *X, const int incX,
                 double *Y, const int incY)
{
  __m128d v1 ;
  register unsigned int i ;
  double copy [2] __attribute__ ((aligned (16))) ;
  #pragma omp parallel for private(v1, copy)
  for (i = 0; i < N; i += 2){
      v1 = _mm_load_pd (X+i) ;

      _mm_store_pd (copy, v1) ;
      Y [i] = copy [0] ;
      Y [i+1] = copy [1] ;
    }
}

void mncblas_ccopy(const int N, const void *X, const int incX,
		                    void *Y, const int incY)
{
  __m128 v1 ;
  register unsigned int i ;
  float copy [4] __attribute__ ((aligned (16))) ;
  complexe_float_t *A = (complexe_float_t*) X;
  complexe_float_t *B = (complexe_float_t*) Y;

  #pragma omp parallel for private(v1, copy)
  for (i =0; i < N; i += 2){
    v1 = _mm_load_ps (&(A[i].real)) ;

    _mm_store_ps (copy, v1) ;

    B [i].real = copy [0] ;
    B [i].imaginary = copy [1] ;
    B [i+1].real = copy [2] ;
    B [i+1].imaginary = copy [3] ;
  }
}

void mncblas_zcopy(const int N, const void *X, const int incX,
		                    void *Y, const int incY)
{
  __m128d v1 ;
  register unsigned int i ;
  double copy [2] __attribute__ ((aligned (16))) ;
  complexe_double_t *A = (complexe_double_t*) X;
  complexe_double_t *B = (complexe_double_t*) Y;
  
  #pragma omp parallel for private(v1, copy)
  for (i =0; i < N; i += 1){

    v1 = _mm_load_pd (&A[i].real) ;

    _mm_store_pd (copy, v1) ;

    B [i].real = copy [0] ;
    B [i].imaginary = copy [1] ;

    B[i].real = A[i].real;
    B[i].imaginary = A[i].imaginary;
  }
}
