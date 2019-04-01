#include "mnblas.h"
#include "complexe.h"

void mnblas_saxpy(const int N, const float alpha, const float *X,
                 const int incX, float *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  for (; ((i < N) && (j < N)) ; i += incX, j+=incY)
    {
      Y[j] = alpha*X[i] + Y[j];
    }

    return;

}

void mnblas_daxpy(const int N, const double alpha, const double *X,
                 const int incX, double *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  for (; ((i < N) && (j < N)) ; i += incX, j+=incY)
    {
      Y[j] = alpha*X[i] + Y[j];
    }

    return;

}

void mnblas_caxpy(const int N, const void *alpha, const void *X,
                 const int incX, void *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  complexe_float_t* x = (complexe_float_t*)X;
  complexe_float_t* y = (complexe_float_t*)Y;
  complexe_float_t* a = (complexe_float_t*)alpha;
  for (; ((i < N) && (j < N)) ; i += incX, j+=incY)
      y[j] = add_complexe_float(mult_complexe_float(x[i], *a),y[j]);
}

void mnblas_zaxpy(const int N, const void *alpha, const void *X,
                 const int incX, void *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  complexe_double_t* x = (complexe_double_t*)X;
  complexe_double_t* y = (complexe_double_t*)Y;
  complexe_double_t* a = (complexe_double_t*)alpha;
  for (; ((i < N) && (j < N)) ; i += incX, j+=incY)
        y[j] = add_complexe_double(mult_complexe_double(x[i], *a),y[j]);
}
