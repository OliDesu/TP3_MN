#include <stdio.h>
#include <x86intrin.h>
#include "mnblas.h"
#include "complexe.h"
#include "flop.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define VECSIZE   64

#define NB_FOIS    5

typedef float vfloat [VECSIZE] ;
typedef double vdouble [VECSIZE] ;
typedef complexe_float_t vcfloat [VECSIZE] ;
typedef complexe_double_t vcdouble [VECSIZE] ;

typedef float mfloat [VECSIZE][VECSIZE] ;
typedef double mdouble [VECSIZE][VECSIZE] ;
typedef complexe_float_t mcfloat [VECSIZE][VECSIZE] ;
typedef complexe_double_t mcdouble [VECSIZE][VECSIZE] ;

mfloat af, bf, cf;
mdouble ad, bd, cd;
mcfloat acf, bcf, ccf;
mcdouble acd, bcd, ccd;

void vector_init (vfloat V, float x, int taille)
{
  register unsigned int i ;
  for (i = 0; i < taille; i++)
    V [i] = x ;

  return ;
}
void mat_init (mfloat V, float x, int row)
{
  for (int i = 0; i < row; i++)
    for(int j = 0; j < row; j++)
      V [i][j] = x ;
  return ;
}

void vector_double_init (vdouble V, double x, int taille)
{
  register unsigned int i ;

  for (i = 0; i < taille; i++)
    V [i] = x ;

  return ;
}
void mat_double_init (mdouble V, double x, int row)
{
  for (int i = 0; i < row; i++)
    for(int j = 0; j < row; j++)
      V [i][j] = x ;
  return ;
}

void vector_c_float_init (vcfloat V, float x1, float x2, int taille)
{
  register unsigned int i ;

  for (i = 0; i < taille; i++) {
    V [i].real = x1 ;
    V [i].imaginary = x2 ;
  }
  return ;
}
void mat_c_float_init (mcfloat V, float x1, float x2, int row)
{
  for (int i = 0; i < row; i++)
    for(int j = 0; j < row; j++) {
      V [i][j].real = x1 ;
      V [i][j].imaginary = x2 ;
    }
  return ;
}

void vector_c_double_init (vcdouble V, double x1, double x2, int taille)
{
  register unsigned int i ;

  for (i = 0; i < taille; i++) {
    V [i].real = x1 ;
    V [i].imaginary = x2 ;
  }
  return ;
}
void mat_c_double_init (mcdouble V, double x1, double x2, int row)
{
  for (int i = 0; i < row; i++)
    for(int j = 0; j < row; j++) {
      V [i][j].real = x1 ;
      V [i][j].imaginary = x2 ;
    }
  return ;
}

void vector_printf (vfloat V, int taille)
{
  register unsigned int i ;

  for (i = 0; i < taille; i++)
    printf ("%f \t", V[i]) ;
  printf ("\n") ;
  
  return ;
}

void vector_printd (vdouble V, int taille)
{
  register unsigned int i ;

  for (i = 0; i < taille; i++)
    printf ("%lf \t", V[i]) ;
  printf ("\n") ;
  
  return ;
}

void matrice_printcf (mcfloat V, int taille)
{
  register unsigned int i ;

  for (i = 0; i < taille; i++)
    for(size_t j = 0; j < taille; j++) 
      printf ("%f+i*%f \t", V[i][j].real, V[i][j].imaginary) ;
  printf ("\n") ;
  
  return ;
}

int main (int argc, char **argv)
{
  unsigned long long start, end ;
  int i ;
  srand(time(NULL));


  printf("\n\n========================GEMM FLOAT=========================== \n");
  for (i = 0 ; i < NB_FOIS; i++)
  {
    printf("TEST n°%d \n", i);
    mat_init (af, 1.0, VECSIZE) ;
    mat_init (bf, 1.0, VECSIZE) ;
    mat_init (cf, 32.0, VECSIZE) ;
    start = _rdtsc () ;
        mncblas_sgemm (100, 100, 100, VECSIZE,VECSIZE,VECSIZE, 1.0, (float*)af, 1, (float*)bf, 1, 1.0, (float*)cf, 1) ;
    end = _rdtsc () ;
    
    printf ("mncblas_sgemm %d : nombre de cycles: %Ld \n", i, end-start) ;
    calcul_flop ("sgemm ",(VECSIZE*VECSIZE*3 +VECSIZE*VECSIZE*VECSIZE*2), end-start) ;
    //vector_printf(resf);  
  }

  printf("\n\n========================GEMM DOUBLE=========================== \n");
  for (i = 0 ; i < NB_FOIS; i++)
  {
    printf("TEST n°%d \n", i);
    mat_double_init (ad, 1.0, VECSIZE) ;
    mat_double_init (bd, 1.0, VECSIZE) ;
    mat_double_init (cd, 32.0, VECSIZE) ;
    start = _rdtsc () ;
        mncblas_dgemm (100, 100, 100, VECSIZE,VECSIZE,VECSIZE, 1.0, (double*)ad, 1, (double*)bd, 1, 1.0, (double*)cd, 1) ;
    end = _rdtsc () ;
    
    printf ("mncblas_dgemm %d : nombre de cycles: %Ld \n", i, end-start) ;
    calcul_flop ("dgemm ",(VECSIZE*VECSIZE*3 +VECSIZE*VECSIZE*VECSIZE*2), end-start) ;
    //vector_printf(resf);
  }

  printf("\n\n========================GEMM COMPLEXEF=========================== \n");
  for (i = 0 ; i < NB_FOIS; i++)
  {
    printf("TEST n°%d \n", i);
    mat_c_float_init(acf, 1.0, 1.0, VECSIZE);
    mat_c_float_init(bcf, 1.0, 0.0, VECSIZE);
    mat_c_float_init(ccf, 0.0, 1.0, VECSIZE);
    complexe_float_t alpha,beta;
    alpha.real = 1.0;
    alpha.imaginary = 0.0;
    beta.real = 1.0;
    beta.imaginary = 0.0;
    start = _rdtsc();
      mncblas_cgemm(MNCblasRowMajor, MNCblasNoTrans, MNCblasNoTrans, VECSIZE, VECSIZE, VECSIZE, &alpha, acf, 1, bcf, 1, &beta, ccf, 1);
    end = _rdtsc();
    printf("mncblas_cgemv nombre de cycles: %Ld \n", end - start);
    calcul_flop("cfgemv ",(VECSIZE * VECSIZE * VECSIZE * 8) + (VECSIZE * VECSIZE *14), end - start);
  }
  printf("Resultat cfloat : %f + i %f \n", ccf[0][0].real, ccf[0][0].imaginary); 
  printf("Resultat cfloat : %f + i %f \n", ccf[0][1].real, ccf[0][1].imaginary); 
  //matrice_printcf(ccf, VECSIZE);

   printf("\n\n========================GEMM COMPLEXED=========================== \n");
    for (i = 0 ; i < NB_FOIS; i++)
  {
    printf("TEST n°%d \n", i);
    mat_c_double_init(acd, 1.0, 1.0, VECSIZE);
    mat_c_double_init(bcd, 1.0, 0.0, VECSIZE);
    mat_c_double_init(ccd, 0.0, 1.0, VECSIZE);
    complexe_double_t alphad,betad;
    alphad.real = 1.0;
    alphad.imaginary = 0.0;
    betad.real = 1.0;
    betad.imaginary = 0.0;
    start = _rdtsc();
      mncblas_zgemm(MNCblasRowMajor, MNCblasNoTrans, MNCblasNoTrans, VECSIZE, VECSIZE, VECSIZE, &alphad, acd, 1, bcd, 1, &betad, ccd, 1);
    end = _rdtsc();
    printf("mncblas_cgemv nombre de cycles: %Ld \n", end - start);
    calcul_flop("cfgemv ",(VECSIZE * VECSIZE * VECSIZE * 8) + (VECSIZE * VECSIZE *14), end - start);
  }
  printf("Resultat cfloat : %f + i %f \n", ccd[0][0].real, ccd[0][0].imaginary); 
  printf("Resultat cfloat : %f + i %f \n", ccd[0][1].real, ccd[0][1].imaginary); 
  //vector_printf(yd, VECSIZE);
}