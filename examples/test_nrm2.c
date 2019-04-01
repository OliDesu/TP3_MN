#include <stdio.h>
#include <x86intrin.h>

#include "mnblas.h"
#include "complexe.h"

#include "flop.h"

#define VECSIZE    1024

#define NB_FOIS    5

typedef float vfloat [VECSIZE] ;
typedef double vdouble [VECSIZE] ;
typedef complexe_float_t vcfloat [VECSIZE] ;
typedef complexe_double_t vcdouble [VECSIZE] ;

vfloat vec1, vec2 ;
vdouble vecd1, vecd2 ;
vcfloat veccf1, veccf2 ;
vcdouble veccd1, veccd2 ;

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
    printf ("%f ", V[i]) ;
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
  float resf;double resd;
  printf("\n\n========================NRM2 FLOAT=========================== \n");
  printf("Calcul testé (vec1[i] == 1.) : (somme pour i allant de 0 a 1024) vec1[i] = 1024.\n\n");
  for (i = 0 ; i < NB_FOIS; i++)
  {
    printf("TEST n°%d \n", i);
    vector_init (vec1, 1.0) ;

    start = _rdtsc () ;
        resf =  mnblas_snrm2 (VECSIZE, vec1, 1) ;
    end = _rdtsc () ;
    
    printf ("mnblas_saxpy %d : nombre de cycles: %Ld \n", i, end-start) ;
    calcul_flop ("sdot ", 2 * VECSIZE, end-start) ;
    printf("Resultat float : %f \n", resf);    
  }

  printf("\n\n========================NRM2 DOUBLE=========================== \n");
  printf("Calcul testé (vec1[i] == 1.) : (somme pour i allant de 0 a 1024) vec1[i] = 1024.\n\n");
  for (i = 0 ; i < NB_FOIS; i++)
  {
    printf("TEST n°%d \n", i);
    vector_double_init (vecd1, 1.0) ;

    start = _rdtsc () ;
      resd =  mnblas_dnrm2 (VECSIZE, vecd1, 1) ;
    end = _rdtsc () ;
    
    printf ("mnblas_saxpy %d : nombre de cycles: %Ld \n", i, end-start) ;
    calcul_flop ("sdot ", 2 * VECSIZE, end-start) ;
    printf("Resultat double : %f \n", resd);    
  }

  printf("\n\n========================NRM2 COMPLEXEF=========================== \n");
  printf("Calcul testé (vec1[i].real == 1., vec1[i].imaginary == 1.) : (somme pour i allant de 0 a 1024) |vec1[i].real| + |vec1[i].imaginary| = 2048.\n\n");
  for (i = 0 ; i < NB_FOIS; i++)
  {
    printf("TEST n°%d \n", i);
    vector_c_float_init (veccf1, 1.0, 1.0) ;
    start = _rdtsc () ;
      resf = mnblas_scnrm2 (VECSIZE, veccf1, 1) ;
    end = _rdtsc () ;
    
    printf ("mncblas_sdot %d : nombre de cycles: %Ld \n", i, end-start) ;
    calcul_flop ("sdot ", 2 * VECSIZE, end-start) ;
    printf("Resultat float: %f \n", resf);
  }

   printf("\n\n========================NRM2 COMPLEXED=========================== \n");
  printf("Calcul testé (vec1[i].real == 1., vec1[i].imaginary == 1.) : (somme pour i allant de 0 a 1024) |vec1[i].real| + |vec1[i].imaginary| = 2048.\n\n");
  for (i = 0 ; i < NB_FOIS; i++)
  {
    printf("TEST n°%d \n", i);
    vector_c_double_init (veccd1, 1.0, 0.0) ;
    start = _rdtsc () ;
      resd = mnblas_dznrm2 (VECSIZE, veccd1, 1) ;
    end = _rdtsc () ;
    
    printf ("mncblas_sdot %d : nombre de cycles: %Ld \n", i, end-start) ;
    calcul_flop ("sdot ", 2 * VECSIZE, end-start) ;
    printf("Resultat float: %f \n", resd);
  }
}
