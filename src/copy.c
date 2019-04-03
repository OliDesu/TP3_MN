#include "../include/mnblas.h"
#include "../include/complexe.h"
void mncblas_scopy(const int N, const float *X, const int incX,
                 float *Y, const int incY)
{

  #pragma omp parallel for
  for (int i = 0; i < N; i += incX)
    {
      Y [i] = X [i] ;
    }

  return ;
}

void mncblas_dcopy(const int N, const double *X, const int incX,
                 double *Y, const int incY)
{
  #pragma omp parallel for
  for (int i = 0; i < N; i += incX){
      Y[i] = X[i];
    }
}

void mncblas_ccopy(const int N, const void *X, const int incX,
		                    void *Y, const int incY)
{
  complexe_float_t *A = (complexe_float_t*) X;
  complexe_float_t *B = (complexe_float_t*) Y;

  #pragma omp parallel for
  for (int i =0; i < N; i += incX){
    B[i].real = A[i].real;
    B[i].imaginary = A[i].imaginary;
  }
}

void mncblas_zcopy(const int N, const void *X, const int incX,
		                    void *Y, const int incY)
{
  complexe_double_t *A = (complexe_double_t*) X;
  complexe_double_t *B = (complexe_double_t*) Y;
  
  #pragma omp parallel for
  for (int i =0; i < N; i += incX){
    B[i].real = A[i].real;
    B[i].imaginary = A[i].imaginary;
  }
}
