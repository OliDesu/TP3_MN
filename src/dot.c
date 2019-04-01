#include "mnblas.h"
#include "complexe.h"

float mncblas_sdot(const int N, const float *X, const int incX,
                 const float *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  register float dot = 0.0 ;

  for (; ((i < N) && (j < N)) ; i += incX, j+=incY)
    {
      dot = dot + X [i] * Y [j] ;
    }

  return dot ;
}

double mncblas_ddot(const int N, const double *X, const int incX,
                 const double *Y, const int incY)
{
  register unsigned int i = 0;
  register unsigned int j = 0;
  register double dot = 0.0;
  for (; i < N && j < N; i += incX, j+=incY) {
    dot = dot + X [i] * Y [j] ;
  }

  return dot;
}

void   mncblas_cdotu_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotu)
{
  register unsigned int i = 0;
  register unsigned int j = 0;
  register complexe_float_t* dot = (complexe_float_t*)dotu;
  complexe_float_t* x = (complexe_float_t*)X;
  complexe_float_t* y = (complexe_float_t*)Y;
  for (; i < N && j < N; i += incX, j+=incY)
    *dot = add_complexe_float(*dot, mult_complexe_float(x[i], y[j]));

}

void   mncblas_cdotc_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotc)
{
  register unsigned int i = 0;
  register unsigned int j = 0;
  register complexe_float_t* dot = dotc;
  complexe_float_t* x = (complexe_float_t*)X;
  complexe_float_t* y = (complexe_float_t*)Y;
  for (; i < N && j < N; i += incX, j+=incY)
      *dot = add_complexe_float(*dot, mult_complexe_float(conj_complexe_float(x[i]), y[j]));
}

void   mncblas_zdotu_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotu)
{
  register unsigned int i = 0;
  register unsigned int j = 0;
  register complexe_double_t* dot = dotu;
  complexe_double_t* x = (complexe_double_t*)X;
  complexe_double_t* y = (complexe_double_t*)Y;
  for (; i < N && j < N; i += incX, j+=incY)
    *dot = add_complexe_double(*dot, mult_complexe_double(x[i], y[j]));
}

void   mncblas_zdotc_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotc)
{
  register unsigned int i = 0;
  register unsigned int j = 0;
  register complexe_double_t* dot = dotc;
  complexe_double_t* x = (complexe_double_t*)X;
  complexe_double_t* y = (complexe_double_t*)Y;
  for (; i < N && j < N; i += incX, j+=incY)
    *dot = add_complexe_double(*dot, mult_complexe_double(conj_complexe_double(x[i]), y[j]));
}
