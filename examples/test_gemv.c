#include <stdio.h>
#include <x86intrin.h>
#include "mnblas.h"
#include "complexe.h"
#include "flop.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define VECSIZE    512

#define NB_FOIS    5

typedef float vfloat [VECSIZE] ;
typedef double vdouble [VECSIZE] ;
typedef complexe_float_t vcfloat [VECSIZE] ;
typedef complexe_double_t vcdouble [VECSIZE] ;

vfloat vec1, vec2, resf ;
vdouble vecd1, vecd2, resd ;
vcfloat veccf1, veccf2, rescf ;
vcdouble veccd1, veccd2, rescd ;

void vector_init (vfloat V, float x)
{
  register unsigned int i ;
  for (i = 0; i < VECSIZE; i++)
    V [i] = x ;

  return ;
}
void vector_double_init (vdouble V, double x)
{
  register unsigned int i ;

  for (i = 0; i < VECSIZE; i++)
    V [i] = x ;

  return ;
}
void vector_c_float_init (vcfloat V, float x1, float x2)
{
  register unsigned int i ;

  for (i = 0; i < VECSIZE; i++) {
    V [i].real = x1 ;
    V [i].imaginary = x2 ;
  }
  return ;
}

void vector_c_double_init (vcdouble V, double x1, double x2)
{
  register unsigned int i ;

  for (i = 0; i < VECSIZE; i++) {
    V [i].real = x1 ;
    V [i].imaginary = x2 ;
  }
  return ;
}

void vector_printf (vfloat V)
{
  register unsigned int i ;

  for (i = 0; i < VECSIZE; i++)
    printf ("%f \t", V[i]) ;
  printf ("\n") ;
  
  return ;
}

void vector_printd (vdouble V)
{
  register unsigned int i ;

  for (i = 0; i < VECSIZE; i++)
    printf ("%lf ", V[i]) ;
  printf ("\n") ;
  
  return ;
}

int main (int argc, char **argv)
{
  unsigned long long start, end ;
  int i ;
  srand(time(NULL));

  printf("\n\n========================GEMV FLOAT=========================== \n");
  printf("Calcul testé (vec1[i] == 1.) : (somme pour i allant de 0 a 1024) vec1[i] = 1024.\n\n");
  for (i = 0 ; i < NB_FOIS; i++)
  {
    printf("TEST n°%d \n", i);
    vector_init (vec1, 1.0) ;
    vector_init (vec2, 1.0) ;
    vector_init (resf, 0.0);
    start = _rdtsc () ;
        mncblas_sgemv (100, 100, VECSIZE, VECSIZE, 1.0, vec1, 1, vec2, 1, 1.0,resf, 1) ;
    end = _rdtsc () ;
    
    printf ("mnblas_saxpy %d : nombre de cycles: %Ld \n", i, end-start) ;
    calcul_flop ("sdot ",(VECSIZE * VECSIZE * 2) + (VECSIZE *6), end-start) ;
    printf("Resultat float : %f \n", resf[0]);
    //vector_printf(resf);  
  }

  printf("\n\n========================GEMV DOUBLE=========================== \n");
  printf("Calcul testé (vec1[i] == 1.) : (somme pour i allant de 0 a 1024) vec1[i] = 1024.\n\n");
  for (i = 0 ; i < NB_FOIS; i++)
  {
    printf("TEST n°%d \n", i);
    vector_double_init (vecd1, 1.0) ;
    vector_double_init (vecd2, 1.0) ;
    vector_double_init (resd, 0.0);

    start = _rdtsc () ;
        mncblas_dgemv (100, 100, VECSIZE, VECSIZE, 1.0, vecd1, 1, vecd2, 1, 1.0,resd, 1) ;
    end = _rdtsc () ;
    
    printf ("mnblas_saxpy %d : nombre de cycles: %Ld \n", i, end-start) ;
    calcul_flop ("sdot ",(VECSIZE * VECSIZE * 2) + (VECSIZE *6), end-start) ;
    printf("Resultat double : %f \n", resd[0]);    
  }

  printf("\n\n========================GEMV COMPLEXEF=========================== \n");
	 vector_c_float_init(veccf1, 1.0,1.0);
	 vector_c_float_init(veccf2, 1.0,1.0);
	 vector_c_float_init(rescf, 0.0,0.0);
	complexe_float_t alpha,beta;
  alpha.real = 1.0;
  alpha.imaginary = 1.0;
  beta.real = 1.0;
  beta.imaginary = 1.0;
for (i = 0 ; i < NB_FOIS; i++)
  {
    printf("TEST n°%d \n", i);
    start = _rdtsc();
    mncblas_cgemv(MNCblasRowMajor, MNCblasNoTrans, 100, 100, &alpha, rescf, 3.5, veccf1, 3, &beta, veccf2, 4);
    end = _rdtsc();
    printf("mncblas_gemv nombre de cycles: %Ld \n", end - start);
    calcul_flop("sdot ",(VECSIZE * VECSIZE * 8) + (VECSIZE *14), end - start);
  }
   printf("\n\n========================GEMV COMPLEXED=========================== \n");
   vector_c_double_init(veccd1, 1.0,1.0);
	 vector_c_double_init(veccd2, 1.0,1.0);
	 vector_c_double_init(rescd, 0.0,0.0);
	complexe_double_t alpha1,beta1;
  alpha1.real = 1.0;
  alpha1.imaginary = 1.0;
  beta1.real = 1.0;
  beta1.imaginary = 1.0;
for (i = 0 ; i < NB_FOIS; i++)
  {
    printf("TEST n°%d \n", i);
    start = _rdtsc();
    mncblas_cgemv(MNCblasRowMajor, MNCblasNoTrans, 100, 100, &alpha1, rescd, 3.5, veccd1, 3, &beta1, veccd2, 4);
    end = _rdtsc();
    printf("mncblas_gemv nombre de cycles: %Ld \n", end - start);
    calcul_flop("sdot ",(VECSIZE * VECSIZE * 8) + (VECSIZE *14), end - start);
  }
}