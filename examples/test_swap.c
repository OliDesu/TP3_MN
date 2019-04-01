#include <stdio.h>
#include <x86intrin.h>
#include "mnblas.h"
#include "complexe.h"
#include <math.h>
#include <time.h>
//==========================================================//
//=======================TEST SWAP==========================//
//==========================================================//

#define N 1
int main(int argc, char **argv){
  srand(time(NULL));
  printf("========================SSWAP=========================== \n");
  float X[N];
  float Y[N];
  printf("\n-----AVANT SWAP------\n");
  printf("X = ");
  for (int i = 0 ; i < N ;i++){
    X[i] = rand()%10;
    printf("%f ", X[i]);
  }
  printf("\n");
  printf("Y = ");
  for (int i = 0 ; i < N ;i++){
    Y[i] = rand()%10;
    printf("%f ", Y[i]);
  }
  printf("\n");
  const int incX = 1;
  const int incY = 1;
  mncblas_sswap(N, X, incX,Y,incY);
  printf("\n------APRES SWAP------\n");
  printf("X = ");
   for(int i = 0 ; i < N ; i++){
    printf("%lf ", X[i]);
  }
  printf(" \nY = ");
  for(int i = 0 ; i < N ; i++){
  printf("%lf ", Y[i]);
  }
 

  printf("\n========================DSWAP=========================== \n");
  double X1[N];
  double Y1[N];
  printf("\n------AVANT SWAP------ \n");
  printf("X = ");
  for (int i = 0 ; i < N ;i++){
    X1[i] = rand()%10;
    printf("%lf ", X1[i]);
  }
  printf("\n");
  printf("Y = ");
  for (int i = 0 ; i < N ;i++){
    Y1[i] = rand()%10;
    printf("%lf ", Y1[i]);
  }
  printf("\n");
  mncblas_dswap(N, X1, incX,Y1,incY);
  printf("\n------APRES SWAP------\n");
  printf("X = ");
   for(int i = 0 ; i < N ; i++){
    printf("%lf ", X1[i]);
  }
  printf(" \nY = ");
  for(int i = 0 ; i < N ; i++){
  printf("%lf ", Y1[i]);
  }
  printf("\n");

  printf("========================CSWAP=========================== \n");
  printf("\n------AVANT SWAP------\n");
  complexe_float_t X2;
  complexe_float_t Y2;
  X2.imaginary = rand()%10;
  X2.real =rand()%10;
  Y2.imaginary = rand()%10;
  Y2.real =rand()%10;
  printf("X = %f + %f * i \n", X2.real,X2.imaginary);
  printf("Y = %f + %f * i \n", Y2.real,Y2.imaginary);
  mncblas_cswap(N, &X2, incX,&Y2,incY);
  printf("\n------APRES SWAP------\n");
  printf("X = %f + %f * i \n", X2.real,X2.imaginary);
  printf("Y = %f + %f * i \n", Y2.real,Y2.imaginary);



  printf("========================ZSWAP=========================== \n");

  printf("\n------AVANT SWAP------\n");
  complexe_double_t X3;
  complexe_double_t Y3;
  X3.imaginary = rand()%10;
  X3.real =rand()%10;
  Y3.imaginary = rand()%10;
  Y3.real =rand()%10;
  printf("X = %f + %f * i \n", X3.real,X3.imaginary);
  printf("Y = %f + %f * i \n", Y3.real,Y3.imaginary);
  mncblas_zswap(N, &X3, incX,&Y3,incY);
  printf("\n------APRES SWAP------\n");
  printf("X = %f + %f * i \n", X3.real,X3.imaginary);
  printf("Y = %f + %f * i \n", Y3.real,Y3.imaginary);

}
