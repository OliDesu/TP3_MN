#include "mnblas.h"
#include "complexe.h"
#include <smmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <x86intrin.h>


void mncblas_sgemv(const MNCBLAS_LAYOUT layout,
                 const MNCBLAS_TRANSPOSE TransA, const int M, const int N,
                 const float alpha, const float *A, const int lda,
                 const float *X, const int incX, const float beta,
                 float *Y, const int incY)
{
  __m128 v1res, v2res, vbeta, vY, res ;
  register unsigned int i, k ;
  float gemv [4] __attribute__ ((aligned (16))) ;
  register float res_total = 0.0, res_total2 = 0.0, res_total3 = 0.0, res_total4 = 0.0;
  
  #pragma omp parallel for private(res_total, res_total2, res_total3, res_total4) schedule(static)
  for(i = 0; i < M; i+=4)
  {
    vbeta = _mm_set1_ps(beta);
    vY = _mm_load_ps (Y+i);
    

    res_total = 0; res_total2 = 0; res_total3 = 0; res_total4 = 0;
    for(k = 0; k < N; k+=4)
    {
      v1res = _mm_set1_ps(X[i]);
      v2res = _mm_load_ps (A+(i*N+k)) ;
      res = _mm_dp_ps(v1res, v2res, 0xFF) ;
      _mm_store_ps (gemv, res) ;
      res_total += gemv[0] + gemv[1] + gemv[2] + gemv[3];

      v1res = _mm_set1_ps(X[i+1]);
      v2res = _mm_load_ps (A+((i+1)*N+k)) ;
      res = _mm_dp_ps(v1res, v2res, 0xFF) ;
      _mm_store_ps (gemv, res) ;
      res_total2 += gemv[0] + gemv[1] + gemv[2] + gemv[3];

      v1res = _mm_set1_ps(X[i+2]);
      v2res = _mm_load_ps (A+((i+2)*N+k)) ;
      res = _mm_dp_ps(v1res, v2res, 0xFF) ;
      _mm_store_ps (gemv, res) ;
      res_total3 += gemv[0] + gemv[1] + gemv[2] + gemv[3];

      v1res = _mm_set1_ps(X[i+3]);
      v2res = _mm_load_ps (A+((i+3)*N+k)) ;
      res = _mm_dp_ps(v1res, v2res, 0xFF) ;
      _mm_store_ps (gemv, res) ;
      res_total4 += gemv[0] + gemv[1] + gemv[2] + gemv[3];
    }

    res = _mm_dp_ps(vbeta, vY, 0xFF) ;
    _mm_store_ps (gemv, res) ;
    
    Y[i] = alpha * res_total + gemv[0];
    Y[i] = alpha * res_total2 + gemv[1];
    Y[i] = alpha * res_total3 + gemv[2];
    Y[i+1] = alpha * res_total4 + gemv[3];
  }
}

void mncblas_dgemv(MNCBLAS_LAYOUT layout,
                 MNCBLAS_TRANSPOSE TransA, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 const double *X, const int incX, const double beta,
                 double *Y, const int incY)
{
  __m128d v1res, v2res, vbeta, vY, res ;
  register unsigned int i, k ;
  double gemv [2] __attribute__ ((aligned (16))) ;
  register double res_total = 0.0, res_total2 = 0.0;
  
  #pragma omp parallel for private(res_total, res_total2) schedule(static)
  for(i = 0; i < M; i+=2)
  {
    vbeta = _mm_set1_pd(beta);
    vY = _mm_load_pd (Y+i);
    

    res_total = 0; res_total2 = 0;
    for(k = 0; k < N; k+=2)
    {
      v1res = _mm_set1_pd(X[i]);
      v2res = _mm_load_pd (A+(i*N+k)) ;
      res = _mm_dp_pd(v1res, v2res, 0xFF) ;
      _mm_store_pd (gemv, res) ;
      res_total += gemv[0] + gemv[1];

      v1res = _mm_set1_pd(X[i+1]);
      v2res = _mm_load_pd (A+((i+1)*N+k)) ;
      res = _mm_dp_pd(v1res, v2res, 0xFF) ;
      _mm_store_pd (gemv, res) ;
      res_total2 += gemv[0] + gemv[1];
    }

    res = _mm_dp_pd(vbeta, vY, 0xFF) ;
    _mm_store_pd (gemv, res) ;
    
    Y[i] = alpha * res_total + gemv[0];
    Y[i+1] = alpha * res_total2 + gemv[1];
  }
}

void mncblas_cgemv(MNCBLAS_LAYOUT layout,
                 MNCBLAS_TRANSPOSE TransA, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 const void *X, const int incX, const void *beta,
                 void *Y, const int incY)
{
  register unsigned int i ;
  // __m128 v1R, v2R, v1I, v2I;
  // __m128i addr = _mm_set_epi32(8*3, 8*2, 8*1, 8*0);
  // float gemv_r [4] __attribute__ ((aligned (16))) ;
  // float gemv_i [4] __attribute__ ((aligned (16))) ;

  complexe_float_t* a = (complexe_float_t*)A;
  complexe_float_t* x = (complexe_float_t*)X;
  complexe_float_t* y = (complexe_float_t*)Y;
  complexe_float_t* ALPHA = (complexe_float_t*)alpha;
  complexe_float_t* BETA = (complexe_float_t*)beta;
  complexe_float_t res1, res2;

  #pragma omp parallel for private(res1, res2)
  for(i = 0; i < M; i+=2)
  {
    // v1R = _mm_i32gather_ps((&(x[i].real)), addr, 1);
    // v2R = _mm_i32gather_ps((&(y[i].real)), addr, 1);

    // v1I = _mm_i32gather_ps((&(x[i].imaginary)), addr, 1);
    // v2I = _mm_i32gather_ps((&(y[i].imaginary)), addr, 1);

    res1.real = 0.0; res1.imaginary = 0.0;
    res2.real = 0.0; res2.imaginary = 0.0;
    for(int k = 0; k < N; k++)
    {

      res1 =add_complexe_float(res1, mult_complexe_float(a[i*N + k], x[i]));
      res2 =add_complexe_float(res2, mult_complexe_float(a[(i+1)*N + k], x[i+1]));

    }
    y[i] = add_complexe_float(mult_complexe_float(*ALPHA, res1), mult_complexe_float(*BETA, y[i]));
    y[i+1] = add_complexe_float(mult_complexe_float(*ALPHA, res2), mult_complexe_float(*BETA, y[i+1]));
  }
}

void mncblas_zgemv(MNCBLAS_LAYOUT layout,
                 MNCBLAS_TRANSPOSE TransA, const int M, const int N,
                 const void *alpha, const void *A, const int lda,
                 const void *X, const int incX, const void *beta,
                 void *Y, const int incY)
{
  complexe_double_t* a = (complexe_double_t*)A;
  complexe_double_t* x = (complexe_double_t*)X;
  complexe_double_t* y = (complexe_double_t*)Y;
  complexe_double_t* ALPHA = (complexe_double_t*)alpha;
  complexe_double_t* BETA = (complexe_double_t*)beta;
  complexe_double_t res;

  #pragma omp parallel for private(res)
  for(int i = 0; i < M; i+=incX)
  {
    res.real = 0.0;
    res.imaginary = 0.0;
    for(int k = 0; k < N; k++)
    {
      res =add_complexe_double(res, mult_complexe_double(a[i*N + k], x[i]));
    }
    y[i] = add_complexe_double(mult_complexe_double(*ALPHA, res), mult_complexe_double(*BETA, y[i]));
  }
}
