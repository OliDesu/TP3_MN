#include "../include/mnblas.h"
#include "../include/complexe.h"
#include <stdio.h>

void mncblas_sswap(const int N, float *X, const int incX,
                 float *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  register float save ;

  for (; ((i < N) && (j < N)) ; i += incX, j+=incY)
    {
      save = Y [j] ;
      Y [j] = X [i] ;
      X [i] = save ;
    }

  return ;
}

void mncblas_dswap(const int N, double *X, const int incX,
                 double *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  register double save ;

  for (; ((i < N) && (j < N)) ; i += incX, j+=incY)
    {
      save = Y [j] ;
      Y [j] = X [i] ;
      X [i] = save ;
    }

  return ;
}

void mncblas_cswap(const int N, void *X, const int incX,
		                    void *Y, const int incY)
{
  register unsigned int i = 0 ;
  register unsigned int j = 0 ;
  register complexe_float_t save ;
  complexe_float_t* A = X;
  complexe_float_t* B = Y;

  for (; ((i < N) && (j < N)) ; i += incX, j += incY)
    {
      save.real = B[j].real;
      save.imaginary = B[j].imaginary;
      B[j].real = A[i].real;
      B[j].imaginary = A[i].imaginary;

      A[i].real= save.real ;
      A[i].imaginary= save.imaginary;
    }

  return ;
}

void mncblas_zswap(const int N, void *X, const int incX,
		                    void *Y, const int incY)
{

  register unsigned int i = 0 ;
  register unsigned int j = 0 ;

  register complexe_double_t save;

  complexe_double_t* A = (complexe_double_t*) X;
  complexe_double_t* B = (complexe_double_t*) Y;

  for (; ((i < N) && (j < N)) ; i += incX, j += incY)
    {
      save.real = B[j].real;
      save.imaginary = B[j].imaginary;

      B[j].real = A[i].real;
      B[j].imaginary= A[i].imaginary;

      A[i].real= save.real ;
      A[i].imaginary= save.imaginary;


    }


  return ;
}
