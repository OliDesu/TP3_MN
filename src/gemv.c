#include "mnblas.h"
#include "complexe.h"



void mncblas_sgemv(const MNCBLAS_LAYOUT layout,
                 const MNCBLAS_TRANSPOSE TransA, const int M, const int N,
                 const float alpha, const float *A, const int lda,
                 const float *X, const int incX, const float beta,
                 float *Y, const int incY)
{
  register float res = 0.0;
  
  #pragma omp parallel for private(res) schedule(static)
  for(int i = 0; i < M; i+=incX)
  {
    res = 0;
    for(int k = 0; k < N; k++)
    {
      res += A[i*N + k] * X[i];
    }
    Y[i] = alpha * res + beta * Y[i];
  }
}

void mncblas_dgemv(MNCBLAS_LAYOUT layout,
                 MNCBLAS_TRANSPOSE TransA, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 const double *X, const int incX, const double beta,
                 double *Y, const int incY)
{
  register double res = 0.0;

  #pragma omp parallel for private(res)
  for(int i = 0; i < M; i+=incX)
  {
    res = 0.0;
    for(int k = 0; k < N; k++)
    {
      res += A[i*N + k] * X[i];
    }
    Y[i] = alpha * res + beta * Y[i];
  }
}

void mncblas_cgemv(MNCBLAS_LAYOUT layout,
                 MNCBLAS_TRANSPOSE TransA, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 const void *X, const int incX, const void *beta,
                 void *Y, const int incY)
{
  complexe_float_t* a = (complexe_float_t*)A;
  complexe_float_t* x = (complexe_float_t*)X;
  complexe_float_t* y = (complexe_float_t*)Y;
  complexe_float_t* ALPHA = (complexe_float_t*)alpha;
  complexe_float_t* BETA = (complexe_float_t*)beta;
  complexe_float_t res;

  #pragma omp parallel for private(res)
  for(int i = 0; i < M; i+=incX)
  {
    res.real = 0.0;
    res.imaginary = 0.0;
    for(int k = 0; k < N; k++)
    {
      res =add_complexe_float(res, mult_complexe_float(a[i*N + k], x[i]));

    }
    y[i] = add_complexe_float(mult_complexe_float(*ALPHA, res), mult_complexe_float(*BETA, y[i]));
  }
}

void mncblas_zgemv(MNCBLAS_LAYOUT layout,
                 MNCBLAS_TRANSPOSE TransA, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 const void *X, const int incX, const void *beta,
                 void *Y, const int incY)
{
  complexe_double_t* a = (complexe_double_t*)A;
  complexe_double_t* x = (complexe_double_t*)X;
  complexe_double_t* y = (complexe_double_t*)Y;
  complexe_double_t* ALPHA = (complexe_double_t*)alpha;
  complexe_double_t* BETA = (complexe_double_t*)beta;
  complexe_double_t res;

  #pragma omp parallel for private(res)
  for(int i = 0; i < M; i+=incX)
  {
    res.real = 0.0;
    res.imaginary = 0.0;
    for(int k = 0; k < N; k++)
    {
      res =add_complexe_double(res, mult_complexe_double(a[i*N + k], x[i]));
    }
    y[i] = add_complexe_double(mult_complexe_double(*ALPHA, res), mult_complexe_double(*BETA, y[i]));
  }
}
