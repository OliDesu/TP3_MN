#include <stdio.h>
#include <x86intrin.h>
#include "mnblas.h"
#include "complexe.h"
#include <math.h>
//==========================================================//
//=======================TEST MAX===========================//
//==========================================================//

#define N 3
int main(int argc, char **argv){
  printf("========================SMAX=========================== \n");
  float X[N];
  for (int i = 0 ; i < N ;i++){
    X[i] = rand()%10;
  }
  const int incX = 1;
  CBLAS_INDEX c = mnblas_isamax(N, X, incX);
  
  printf("X = ");
   for(int i = 0 ; i < N ; i++){
    printf(" %lf ", X[i]);
  }
  printf("Le max de X est : %lf", X[c]);
  


  printf("\n========================DMAX=========================== \n");
 double X1[N];
  for (int i = 0 ; i < N ;i++){
    X1[i] = rand()%10;
  }

  c = mnblas_idamax(N, X1, incX);
  
  printf("X = ");
   for(int i = 0 ; i < N ; i++){
    printf(" %lf ", X1[i]);
  }
  printf("Le max de X est : %lf", X1[c]);
  

  printf("\n========================CMAX=========================== \n");
  complexe_float_t X2[N];
  for(int i = 0 ; i < N ;i++){
    X2[i].imaginary = rand()%10;
    X2[i].real = rand()%10;
  }

  c = mnblas_icamax(N, X2, incX);
  
  printf("X =\n");
   for(int i = 0 ; i < N ; i++){
     printf("%f + %f * i \n ", X2[i].real, X2[i].imaginary);
  }
  printf("Le max de X est : %lf + %lf *i", X2[c].real, X2[c].imaginary);
  
  


  printf("\n========================ZMAX=========================== \n");
complexe_double_t X3[N];
  for(int i = 0 ; i < N ;i++){
    X3[i].real = rand()%10;
    X3[i].imaginary = rand()%10;
  }

   c = mnblas_izamax(N, X3, incX);
  
  printf("X =\n");
   for(int i = 0 ; i < N ; i++){
    printf("%f + %f * i \n ", X3[i].real, X3[i].imaginary);
  }
  printf("Le max de X est : %lf + %lf *i", X3[c].real, X3[c].imaginary);
  
}
