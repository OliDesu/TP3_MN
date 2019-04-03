#include <stdio.h>
#include <x86intrin.h>
#include "mnblas.h"
#include "complexe.h"
#include "flop.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define MATSIZE    4096
#define VECSIZE    64

#define NB_FOIS    5

typedef float vfloat [VECSIZE] ;
typedef double vdouble [VECSIZE] ;
typedef complexe_float_t vcfloat [VECSIZE] ;
typedef complexe_double_t vcdouble [VECSIZE] ;

typedef float mfloat [VECSIZE][VECSIZE] ;
typedef double mdouble [VECSIZE][VECSIZE] ;
typedef complexe_float_t mcfloat [VECSIZE][VECSIZE] ;
typedef complexe_double_t mcdouble [VECSIZE][VECSIZE] ;

vfloat x, y ; mfloat a;
vdouble xd, yd ; mdouble ad;
vcfloat xcf, ycf ; mcfloat acf;
vcdouble xcd, ycd ; mcdouble acd;

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
    mat_init (a, 1.0, VECSIZE) ;
    vector_init (x, 1.0, VECSIZE) ;
    vector_init (y, 0.0, VECSIZE);
    start = _rdtsc () ;
        mncblas_sgemv (100, 100, VECSIZE, VECSIZE, 1.0, (float*)a, 1, x, 1, 1.0, y, 1) ;
    end = _rdtsc () ;
    
    printf ("mncblas_sgemv %d : nombre de cycles: %Ld \n", i, end-start) ;
    calcul_flop ("fgemv ",(VECSIZE * VECSIZE * 2) + (VECSIZE *4), end-start) ;
    printf("Resultat float : %f \n", y[0]);
    // vector_printf(y, VECSIZE);  
  }

  printf("\n\n========================GEMV DOUBLE=========================== \n");
  printf("Calcul testé (vec1[i] == 1.) : (somme pour i allant de 0 a 1024) vec1[i] = 1024.\n\n");
  for (i = 0 ; i < NB_FOIS; i++)
  {
    printf("TEST n°%d \n", i);
    mat_double_init (ad, 1.0, VECSIZE) ;
    vector_double_init (xd, 1.0, VECSIZE) ;
    vector_double_init (yd, 0.0, VECSIZE) ;

    start = _rdtsc () ;
        mncblas_dgemv (100, 100, VECSIZE, VECSIZE, 1.0, (double*)ad, 1, xd, 1, 1.0,yd, 1) ;
    end = _rdtsc () ;
    
    printf ("mncblas_dgemv %d : nombre de cycles: %Ld \n", i, end-start) ;
    calcul_flop ("dgemv ",(VECSIZE * VECSIZE * 2) + (VECSIZE *6), end-start) ;
    printf("Resultat double : %f \n", yd[0]); 
    //vector_printd(yd, VECSIZE);  
  }

  printf("\n\n========================GEMV COMPLEXEF=========================== \n");
	
  for (i = 0 ; i < NB_FOIS; i++)
  {
    printf("TEST n°%d \n", i);
    mat_c_float_init(acf, 1.0, 1.0, VECSIZE);
    vector_c_float_init(xcf, 1.0, 0.0, VECSIZE);
    vector_c_float_init(ycf, 0.0, 1.0, VECSIZE);
    complexe_float_t alpha,beta;
    alpha.real = 1.0;
    alpha.imaginary = 0.0;
    beta.real = 1.0;
    beta.imaginary = 0.0;
    start = _rdtsc();
      mncblas_cgemv(MNCblasRowMajor, MNCblasNoTrans, VECSIZE, VECSIZE, &alpha, acf, 1, xcf, 1, &beta, ycf, 1);
    end = _rdtsc();
    printf("mncblas_cgemv nombre de cycles: %Ld \n", end - start);
    calcul_flop("cfgemv ",(VECSIZE * VECSIZE * 8) + (VECSIZE *14), end - start);
  }
  printf("Resultat cfloat : %f + i %f \n", ycf[0].real, ycf[0].imaginary); 
  printf("Resultat cfloat : %f + i %f \n", ycf[1].real, ycf[1].imaginary); 
  //vector_printf(yd, VECSIZE);

   printf("\n\n========================GEMV COMPLEXED=========================== \n");
for (i = 0 ; i < NB_FOIS; i++)
  {
    printf("TEST n°%d \n", i);
    mat_c_double_init(acd, 1.0, 1.0, VECSIZE);
    vector_c_double_init(xcd, 1.0, 0.0, VECSIZE);
    vector_c_double_init(ycd, 0.0, 1.0, VECSIZE);
    complexe_double_t alpha,beta;
    alpha.real = 1.0;
    alpha.imaginary = 0.0;
    beta.real = 1.0;
    beta.imaginary = 0.0;
    start = _rdtsc();
      mncblas_zgemv(MNCblasRowMajor, MNCblasNoTrans, VECSIZE, VECSIZE, &alpha, acd, 1, xcd, 1, &beta, ycd, 1);
    end = _rdtsc();
    printf("mncblas_zgemv nombre de cycles: %Ld \n", end - start);
    calcul_flop("cdouble ",(VECSIZE * VECSIZE * 8) + (VECSIZE *14), end - start);
  }
  printf("Resultat cdouble : %lf + i %lf \n", ycd[0].real, ycd[0].imaginary); 
  printf("Resultat cdouble : %lf + i %lf \n", ycd[1].real, ycd[1].imaginary); 
  return 0;
}