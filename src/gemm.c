#include "mnblas.h"
#include "complexe.h"

void mncblas_sgemm(MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
                 MNCBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A,
                 const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc)
{
    register float res = 0.0;

    #pragma omp parallel for private(res)
    for(int i = 0; i < M ; i++)
    {
        for(int k = 0; k < N; k++)
        {
            res=0;
            for(int h = 0; h < N; h++)
            {
                res += A[i*N + h] * B[h*N + i];
                
            }
            C[i * N + k] *= beta;
            C[i * N + k] += alpha * res;
        }
    }
}

void mncblas_dgemm(MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
                 MNCBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc) 
{
    register double res = 0.0;

    #pragma omp parallel for private(res)
    for(int i = 0; i < M ; i++)
    {
        for(int k = 0; k < N; k++)
        {
            res=0;
            for(int h = 0; h < N; h++)
            {
                res += A[i*N + h] * B[h*N + i];
                
            }
            C[i * N + k] *= beta;
            C[i * N + k] += alpha * res;
        }
    }
}

void mncblas_cgemm(MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
                 MNCBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const void *alpha, const void *A,
                 const int lda, const void *B, const int ldb,
                 const void *beta, void *C, const int ldc)
{
    register complexe_float_t res;

    register complexe_float_t * a = (complexe_float_t*) A;
    register complexe_float_t * b = (complexe_float_t*) B;
    register complexe_float_t * c = (complexe_float_t*) C;
    register complexe_float_t * bet = (complexe_float_t*) beta;
    register complexe_float_t * alp = (complexe_float_t*) alpha;

    #pragma omp parallel for private(res)
    for(int i = 0; i < M; i++)
    {
        for(int k = 0; k < N; k++)
        {
            res.real = 0;
            res.imaginary = 0;
            for(int h = 0; h < N; h++)
            {
                res = add_complexe_float(res, mult_complexe_float(a[i*N  + h], b[h*N + i]));
                
            }
            c[i * M + k] = mult_complexe_float(c[i * M + k], *bet);
            c[i * M + k] = add_complexe_float(c[i * M + k], mult_complexe_float(*alp, res));
        }
    }
}

void mncblas_zgemm(MNCBLAS_LAYOUT layout, MNCBLAS_TRANSPOSE TransA,
                 MNCBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const void *alpha, const void *A,
                 const int lda, const void *B, const int ldb,
                 const void *beta, void *C, const int ldc)
{
    register complexe_double_t res;

    register complexe_double_t * a = (complexe_double_t*) A;
    register complexe_double_t * b = (complexe_double_t*) B;
    register complexe_double_t * c = (complexe_double_t*) C;
    register complexe_double_t * bet = (complexe_double_t*) beta;
    register complexe_double_t * alp = (complexe_double_t*) alpha;
    #pragma omp parallel for private(res)
    for(int i = 0; i < M; i++)
    {
        for(int k = 0; k < N; k++)
        {
            res.real = 0;
            res.imaginary = 0;
            for(int h = 0; h < N; h++)
            {
                res = add_complexe_double(res, mult_complexe_double(a[i*N  + h], b[h*N + i]));
                
            }
            c[i * M + k] = mult_complexe_double(c[i * M + k], *bet);
            c[i * M + k] = add_complexe_double(c[i * M + k], mult_complexe_double(*alp, res));
        }
    }
}