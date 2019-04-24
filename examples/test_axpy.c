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

  printf("\n\n========================AXPY FLOAT=========================== \n");
  printf("Calcul testé : 2. * 1. + 2. = 4.\n\n");
  for (i = 0 ; i < NB_FOIS; i++)
    {
      vector_init (vec1, 1.0) ;
      vector_init (vec2, 2.0) ;

      start = _rdtsc () ;
          mnblas_saxpy (VECSIZE, 2.,vec1, 1, vec2, 1) ;
      end = _rdtsc () ;
      
      printf ("mnblas_saxpy %d : nombre de cycles: %Ld \n", i, end-start) ;
      calcul_flop ("sdot ", VECSIZE/4, end-start) ;
      printf("Resultat float : %lf \n", vec2[0]);  
    }

  printf("\n\n========================AXPY DOUBLE=========================== \n");
  printf("Calcul testé : 2. * 1. + 2. = 4.\n\n");
  for (i = 0 ; i < NB_FOIS; i++)
   {
     vector_double_init (vecb1, 1.0) ;
     vector_double_init (vecb2, 2.0) ;

     start = _rdtsc () ;
        mnblas_daxpy (VECSIZE, 2., vecb1, 1, vecb2, 1) ;
     end = _rdtsc () ;
     
     printf ("mnblas_daxpy %d : nombre de cycles: %Ld \n", i, end-start) ;
     calcul_flop ("sdot ", 2 * VECSIZE, end-start) ;
    printf("Resultat double : %lf \n", vecb2[0]);    
   }

  printf("\n\n========================AXPY COMPLEXEF=========================== \n");
  printf("Calcul testé : (2 + 2i) * (1 + i) + (2 + 2i) = 2 + 6i\n\n");
  for (i = 0 ; i < NB_FOIS; i++)
   {
    complexe_float_t* dotu = malloc(sizeof(complexe_float_t));
    dotu->real = 2.0;
    dotu->imaginary = 2.0;
    vector_c_float_init (veccf1, 1.0, 1.0) ;
    vector_c_float_init (veccf2, 2.0, 2.0) ;
    start = _rdtsc () ;
      mnblas_caxpy (VECSIZE, dotu, veccf1, 1, veccf2, 1) ;
    end = _rdtsc () ;
     
    printf ("mncblas_sdot %d : nombre de cycles: %Ld \n", i, end-start) ;
    calcul_flop ("sdot ", 8 * VECSIZE, end-start) ;
    printf("Resultat complexe: %f + %f * i\n", veccf2[0].real, veccf2[0].imaginary);
   }

   printf("\n\n========================AXPY COMPLEXED=========================== \n");
  printf("Calcul testé : (2 + 2i) * (1 + i) + (2 + 2i) = 2 + 6i\n\n");
  for (i = 0 ; i < NB_FOIS; i++)
   {
    complexe_double_t* dotd = malloc(sizeof(complexe_double_t));
    dotd->real = 2.0;
    dotd->imaginary = 2.0;
    vector_c_double_init (veccd1, 1.0, 1.0) ;
    vector_c_double_init (veccd2, 2.0, 2.0) ;
    start = _rdtsc () ;
      mnblas_zaxpy (VECSIZE, dotd, veccd1, 1, veccd2, 1) ;
    end = _rdtsc () ;
     
    printf ("mncblas_sdot %d : nombre de cycles: %Ld \n", i, end-start) ;
    calcul_flop ("sdot ", 8 * VECSIZE, end-start) ;
    printf("Resultat complexe: %f + %f * i\n", veccd2[0].real, veccd2[0].imaginary);
   }
}
