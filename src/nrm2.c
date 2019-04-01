#include "mnblas.h"
#include "complexe.h"
#include <math.h>

float  mnblas_snrm2(const int N, const float *X, const int incX)
{
  register unsigned int i = 0;
  register float somme = 0;
  register float res = 0;

  for (; i < N ; i += incX)
    {
      somme += (float)pow(X[i],2);
    }
    res = (float)sqrt(somme);

    return res;
}

double mnblas_dnrm2(const int N, const double *X, const int incX)
{
  register unsigned int i = 0;
  register double somme = 0;
  register double res = 0;

  for (; i < N ; i += incX)
    {
      somme += (double)pow(X[i],2);
    }
    res = (double)sqrt(somme);

    return res;
}

float  mnblas_scnrm2(const int N, const void *X, const int incX)
{
  register unsigned int i = 0;
  register complexe_float_t somme;
  register complexe_float_t* x = (complexe_float_t*)X;
  register float res = 0;

  somme.real = 0;
  somme.imaginary = 0;

  for (; i < N ; i += incX)
    {
      somme = add_complexe_float(somme,mult_complexe_float(x[i],x[i]));
    }
    res = (float)sqrt(somme.real + somme.imaginary);

    return res;
}

double mnblas_dznrm2(const int N, const void *X, const int incX)
{
  register unsigned int i = 0;
  register complexe_double_t somme;
  register complexe_double_t* x = (complexe_double_t*)X;
  register double res = 0;

  somme.real = 0;
  somme.imaginary = 0;

  for (; i < N ; i += incX)
    {
      somme = add_complexe_double(somme,mult_complexe_double(x[i],x[i]));
    }
    res = (double)sqrt(somme.real + somme.imaginary);

    return res;
}
