#include "mnblas.h"
#include "complexe.h"
#include <smmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <x86intrin.h>

float mncblas_sdot(const int N, const float *X, const int incX,
                 const float *Y, const int incY)
{
  __m128 v1, v2, res ;
  register unsigned int i ;
  float dot [4] __attribute__ ((aligned (16))) ;
  float dot_total = 0.0 ;

  #pragma omp parallel for reduction(+:dot_total)
    for (i = 0; i < N; i = i + 4)
    {
      v1 = _mm_load_ps (X+i) ;
      v2 = _mm_load_ps (Y+i) ;

      res = _mm_dp_ps(v1, v2, 0xFF) ;

      _mm_store_ps (dot, res) ;

      dot_total += dot [0] ;
      dot_total += dot [1] ;
      dot_total += dot [2] ;
      dot_total += dot [3] ;
    }

    return dot_total ;
}

double mncblas_ddot(const int N, const double *X, const int incX,
                 const double *Y, const int incY)
{
  __m128d v1, v2, res ;
  register unsigned int i ;
  double dot [2] __attribute__ ((aligned (16))) ;
  double dot_total = 0.0 ;
  #pragma omp parallel for reduction(+:dot_total)
    for (i = 0; i < N; i += 2)
    {
      v1 = _mm_load_pd (X+i) ;
      v2 = _mm_load_pd (Y+i) ;

      res = _mm_dp_pd (v1, v2, 0xFF) ;

      _mm_store_pd (dot, res) ;

      dot_total += dot [0] ;
      dot_total += dot [1] ;
    }

    return dot_total ;  
}

void   mncblas_cdotu_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotu)
{ 
  __m128 v1R, v2R, v1I, v2I, res1, res2 ;
  register unsigned int i ;
  __m128i addr = _mm_set_epi32(8*3, 8*2, 8*1, 8*0);
  float dot_r [4] __attribute__ ((aligned (16))) ;
  float dot_i [4] __attribute__ ((aligned (16))) ;

  float dot_total_r = 0;
  float dot_total_i = 0;

  register complexe_float_t* dot = (complexe_float_t*)dotu;
  complexe_float_t* x = (complexe_float_t*)X;
  complexe_float_t* y = (complexe_float_t*)Y;

  #pragma omp parallel for reduction(+:dot_total_r) reduction(+:dot_total_i)
  for (i = 0; i < N; i += 4) {
    v1R = _mm_i32gather_ps((&(x[i].real)), addr, 1);
    v2R = _mm_i32gather_ps((&(y[i].real)), addr, 1);

    v1I = _mm_i32gather_ps((&(x[i].imaginary)), addr, 1);
    v2I = _mm_i32gather_ps((&(y[i].imaginary)), addr, 1);

    res1 = _mm_dp_ps (v1R, v2R, 0xFF) ;
    res2 = _mm_dp_ps (v1I, v2I, 0xFF) ;
    res1 = _mm_sub_ps (v1R, v2R) ;
    _mm_store_ps (dot_r, res1) ;
    dot_total_r += dot_r [0] ;
    dot_total_r += dot_r [1] ;
    dot_total_r += dot_r [2] ;
    dot_total_r += dot_r [3] ;

    res1 = _mm_dp_ps (v1R, v2I, 0xFF) ;
    res2 = _mm_dp_ps (v1I, v2R, 0xFF) ;
    res1 = _mm_add_ps (v1R, v2R) ;
    _mm_store_ps (dot_i, res2) ;
    dot_total_i += dot_i [0] ;
    dot_total_i += dot_i [1] ;
    dot_total_i += dot_i [2] ;
    dot_total_i += dot_i [3] ;
  }
  
  dot->real = dot_total_r;
  dot->imaginary = dot_total_i;
}

void   mncblas_cdotc_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotc)
{
  __m128 v1R, v2R, v1I, v2I, res1, res2 ;
  register unsigned int i ;
  __m128i addr = _mm_set_epi32(8*3, 8*2, 8*1, 8*0);
  float dot_r [4] __attribute__ ((aligned (16))) ;
  float dot_i [4] __attribute__ ((aligned (16))) ;

  float dot_total_r = 0;
  float dot_total_i = 0;

  register complexe_float_t* dot = (complexe_float_t*)dotc;
  complexe_float_t* x = (complexe_float_t*)X;
  complexe_float_t* y = (complexe_float_t*)Y;

  #pragma omp parallel for reduction(+:dot_total_r) reduction(+:dot_total_i)
  for (i = 0; i < N; i += 4) {
    v1R = _mm_i32gather_ps((&(x[i].real)), addr, 1);
    v2R = _mm_i32gather_ps((&(y[i].real)), addr, 1);

    v1I = _mm_i32gather_ps((&(x[i].imaginary)), addr, 1);
    v2I = _mm_i32gather_ps((&(y[i].imaginary)), addr, 1);

    res1 = _mm_dp_ps (v1R, v2R, 0xFF) ;
    res2 = _mm_dp_ps (v1I, v2I, 0xFF) ;
    res1 = _mm_add_ps (v1R, v2R) ;
    _mm_store_ps (dot_r, res1) ;
    dot_total_r += dot_r [0] ;
    dot_total_r += dot_r [1] ;
    dot_total_r += dot_r [2] ;
    dot_total_r += dot_r [3] ;

    res1 = _mm_dp_ps (v1R, v2I, 0xFF) ;
    res2 = _mm_dp_ps (v1I, v2R, 0xFF) ;
    res1 = _mm_sub_ps (v1R, v2R) ;
    _mm_store_ps (dot_i, res2) ;
    dot_total_i += dot_i [0] ;
    dot_total_i += dot_i [1] ;
    dot_total_i += dot_i [2] ;
    dot_total_i += dot_i [3] ;
  }
  
  dot->real = dot_total_r;
  dot->imaginary = dot_total_i;
}

void   mncblas_zdotu_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotu)
{
  __m128d v1R, v2R, v1I, v2I, res1, res2 ;
  register unsigned int i ;
  __m128i addr = _mm_set_epi32(16*3, 16*2, 16*1, 16*0);
  double dot_r [2] __attribute__ ((aligned (16))) ;
  double dot_i [2] __attribute__ ((aligned (16))) ;

  double dot_total_r = 0;
  double dot_total_i = 0;

  register complexe_double_t* dot = (complexe_double_t*)dotu;
  complexe_double_t* x = (complexe_double_t*)X;
  complexe_double_t* y = (complexe_double_t*)Y;

  #pragma omp parallel for reduction(+:dot_total_r) reduction(+:dot_total_i)
  for (i = 0; i < N; i += 2) {
    v1R = _mm_i32gather_pd((&(x[i].real)), addr, 1);
    v2R = _mm_i32gather_pd((&(y[i].real)), addr, 1);

    v1I = _mm_i32gather_pd((&(x[i].imaginary)), addr, 1);
    v2I = _mm_i32gather_pd((&(y[i].imaginary)), addr, 1);

    res1 = _mm_dp_pd (v1R, v2R, 0xFF) ;
    res2 = _mm_dp_pd (v1I, v2I, 0xFF) ;
    res1 = _mm_sub_pd (v1R, v2R) ;
    _mm_store_pd (dot_r, res1) ;
    dot_total_r += dot_r [0] ;
    dot_total_r += dot_r [1] ;

    res1 = _mm_dp_pd (v1R, v2I, 0xFF) ;
    res2 = _mm_dp_pd (v1I, v2R, 0xFF) ;
    res1 = _mm_add_pd (v1R, v2R) ;
    _mm_store_pd (dot_i, res2) ;
    dot_total_i += dot_i [0] ;
    dot_total_i += dot_i [1] ;
  }
  
  dot->real = dot_total_r;
  dot->imaginary = dot_total_i;
}

void   mncblas_zdotc_sub(const int N, const void *X, const int incX,
                       const void *Y, const int incY, void *dotc)
{
 __m128d v1R, v2R, v1I, v2I, res1, res2 ;
  register unsigned int i ;
  __m128i addr = _mm_set_epi32(16*3, 16*2, 16*1, 16*0);
  double dot_r [2] __attribute__ ((aligned (16))) ;
  double dot_i [2] __attribute__ ((aligned (16))) ;

  double dot_total_r = 0;
  double dot_total_i = 0;

  register complexe_double_t* dot = (complexe_double_t*)dotc;
  complexe_double_t* x = (complexe_double_t*)X;
  complexe_double_t* y = (complexe_double_t*)Y;

  #pragma omp parallel for reduction(+:dot_total_r) reduction(+:dot_total_i)
  for (i = 0; i < N; i += 2) {
    v1R = _mm_i32gather_pd((&(x[i].real)), addr, 1);
    v2R = _mm_i32gather_pd((&(y[i].real)), addr, 1);

    v1I = _mm_i32gather_pd((&(x[i].imaginary)), addr, 1);
    v2I = _mm_i32gather_pd((&(y[i].imaginary)), addr, 1);

    res1 = _mm_dp_pd (v1R, v2R, 0xFF) ;
    res2 = _mm_dp_pd (v1I, v2I, 0xFF) ;
    res1 = _mm_add_pd (v1R, v2R) ;
    _mm_store_pd (dot_r, res1) ;
    dot_total_r += dot_r [0] ;
    dot_total_r += dot_r [1] ;

    res1 = _mm_dp_pd (v1R, v2I, 0xFF) ;
    res2 = _mm_dp_pd (v1I, v2R, 0xFF) ;
    res1 = _mm_sub_pd (v1R, v2R) ;
    _mm_store_pd (dot_i, res2) ;
    dot_total_i += dot_i [0] ;
    dot_total_i += dot_i [1] ;
  }

  dot->real = dot_total_r;
  dot->imaginary = dot_total_i;
}
