#include <stdio.h>
#include <x86intrin.h>
#include "mnblas.h"
#include "complexe.h"
#include <math.h>
//==========================================================//
//=======================TEST COPY==========================//
//==========================================================//

#define N 3
int main(int argc, char **argv){
  unsigned long long start, end ;

  printf("========================SCOPY=========================== \n");
  float X[N];
  float Y[N];
  for (int i = 0 ; i < N ;i++){
    X[i] = rand()%10;
  }
  const int incX = 1;
  const int incY = 1;
  start = _rdtsc () ;
    mncblas_scopy(N, X, incX,Y,incY);
  end = _rdtsc () ;
  printf("X = ");
   for(int i = 0 ; i < N ; i++){
    printf("%lf ", X[i]);
  }
  printf(" \nY = ");
  for(int i = 0 ; i < N ; i++){
  printf("%lf ", Y[i]);
  }
  printf("\n");
  printf("%lf bit/sec\n", (double)(N*sizeof(float)*1000)/(end - start));


  printf("\n========================DCOPY=========================== \n");
  double X1[N];
  double Y1[N];
  for (int i = 0 ; i < N ;i++){
    X1[i] = rand()%10;
  }
  start = _rdtsc () ;
    mncblas_dcopy(N, X1, incX,Y1,incY);
  end = _rdtsc () ;
  printf("X = ");
   for(int i = 0 ; i < N ; i++){
    printf("%lf ", X1[i]);
  }
  printf(" \nY = ");
  for(int i = 0 ; i < N ; i++){
  printf("%lf ", Y1[i]);
  }
  printf("\n");
  printf("%lf bit/sec\n", (double)(N*sizeof(double)*1000)/(end - start));



  printf("========================CCOPY=========================== \n");

  complexe_float_t X2;
  complexe_float_t Y2;
  X2.imaginary = 1.0;
  X2.real =3.0;
  start = _rdtsc () ;
    mncblas_ccopy(N, &X2, incX,&Y2,incY);
  end = _rdtsc () ;
  printf("X = %f + %f * i \n", X2.real,X2.imaginary);
  printf("Y = %f + %f * i \n", Y2.real,Y2.imaginary);
  printf("%lf bit/sec\n", (double)(sizeof(complexe_float_t)*1000)/(end - start));



  printf("========================ZCOPY=========================== \n");

  complexe_double_t X3;
  complexe_double_t Y3;
  X3.imaginary = 1.0;
  X3.real =3.0;
  start = _rdtsc () ;
    mncblas_ccopy(N, &X3, incX,&Y3,incY);
  end = _rdtsc () ;
  printf("X = %lf + %lf * i \n", X3.real,X3.imaginary);
  printf("Y = %lf + %lf * i \n", Y3.real,Y3.imaginary);
  printf("%lf bit/sec\n", (double)(sizeof(complexe_float_t)*1000)/(end - start));


}
