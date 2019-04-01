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
vdouble vecb1, vecb2 ;
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

void vector_print (vfloat V)
{
  register unsigned int i ;

  for (i = 0; i < VECSIZE; i++)
    printf ("%f ", V[i]) ;
  printf ("\n") ;
  
  return ;
}

int main (int argc, char **argv)
{
 unsigned long long start, end ;
 float res ;
 double resd ;
 int i ;

  printf("\n========================DOT FLOAT=========================== \n");
  for (i = 0 ; i < NB_FOIS; i++)
    {
      vector_init (vec1, 1.0) ;
      vector_init (vec2, 2.0) ;

      start = _rdtsc () ;
          res = mncblas_sdot (VECSIZE, vec1, 1, vec2, 1) ;
      end = _rdtsc () ;
      
      printf ("mncblas_sdot %d : res = %3.2f nombre de cycles: %Ld \n", i, res, end-start) ;
      calcul_flop ("sdot ", 3 * VECSIZE, end-start) ;
      
    }
    printf("\nRésultat : %f", res);

  printf("\n========================DOT DOUBLE=========================== \n");
  for (i = 0 ; i < NB_FOIS; i++)
   {
     vector_double_init (vecb1, 1.0) ;
     vector_double_init (vecb2, 2.0) ;

     start = _rdtsc () ;
        resd = mncblas_ddot (VECSIZE, vecb1, 1, vecb2, 1) ;
     end = _rdtsc () ;
     
     printf ("mncblas_sdot %d : res = %3.2f nombre de cycles: %Ld \n", i, resd, end-start) ;
     calcul_flop ("sdot ", 3 * VECSIZE, end-start) ;
   }
    printf("\nRésultat : %lf", resd);

  printf("\n========================DOT COMPLEXEF=========================== \n");
  
  for (i = 0 ; i < NB_FOIS; i++)
   {
    complexe_float_t* dotu = malloc(sizeof(complexe_float_t));
    dotu->real = 0.0;
    dotu->imaginary = 0.0;
    vector_c_float_init (veccf1, 1.0, 1.0) ;
    vector_c_float_init (veccf2, 2.0, 2.0) ;
    start = _rdtsc () ;
      mncblas_cdotu_sub (VECSIZE, veccf1, 1, veccf2, 1, dotu) ;
    end = _rdtsc () ;
     
    printf ("mncblas_sdot %d : res = %3.2f nombre de cycles: %Ld \n", i, res, end-start) ;
    calcul_flop ("sdot ", 8 * VECSIZE, end-start) ;
    printf("Resultat complexe: %f + %f * i\n", dotu->real, dotu->imaginary);
   }

   printf("\n========================DOT COMPLEXED=========================== \n");
  
  for (i = 0 ; i < NB_FOIS; i++)
   {
    complexe_double_t* dotd = malloc(sizeof(complexe_double_t));
    dotd->real = 0.0;
    dotd->imaginary = 0.0;
    vector_c_double_init (veccd1, 1.0, 1.0) ;
    vector_c_double_init (veccd2, 2.0, 2.0) ;
    start = _rdtsc () ;
      mncblas_zdotu_sub (VECSIZE, veccd1, 1, veccd2, 1, dotd) ;
    end = _rdtsc () ;
     
    printf ("mncblas_sdot %d : res = %3.2f nombre de cycles: %Ld \n", i, res, end-start) ;
    calcul_flop ("sdot ", 8 * VECSIZE, end-start) ;
    printf("Resultat complexe: %f + %f * i\n", dotd->real, dotd->imaginary);

   }
}
