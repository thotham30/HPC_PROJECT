// bcv_svd_timed.c
// BCV Jacobi WITH V accumulation (full SVD).
// Compile: gcc -O3 -march=native -std=c11 bcv_svd_timed.c -o bcv_svd_timed -lm
// Run example: ./bcv_svd_timed 1500 1500 30 4

#define _POSIX_C_SOURCE 200112L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>

double wall_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

#define A_AT(A,m,row,col) ((A)[ (size_t)(col) * (m) + (row) ])

static double *aligned_alloc_d(size_t elems) {
    void *ptr = NULL;
    size_t bytes = elems * sizeof(double);
    if (posix_memalign(&ptr, 64, bytes) != 0) return NULL;
    memset(ptr, 0, bytes);
    return (double*)ptr;
}

static void init_A(double *A, int m, int n) {
    for (int j=0; j<n; ++j)
        for (int i=0; i<m; ++i)
            A_AT(A,m,i,j) = sin((double)(i+1)) * cos((double)(j+1)) + ((i+j)%11) * 0.01;
}

static void init_V_identity(double *V, int n) {
    for (int j=0;j<n;++j)
        for (int i=0;i<n;++i)
            V[(size_t)j * n + i] = (i==j) ? 1.0 : 0.0;
}

/* simple GEMM */
static void dgemm_simple(char opA, char opB,
                         int m, int n, int k,
                         double alpha,
                         const double *A, int lda,
                         const double *B, int ldb,
                         double beta,
                         double *C, int ldc)
{
    for (int jc = 0; jc < n; ++jc) {
        for (int ic = 0; ic < m; ++ic) {
            double sum = 0.0;
            if (opA == 'N' && opB == 'N') {
                for (int l = 0; l < k; ++l)
                    sum += A[(size_t)l * lda + ic] * B[(size_t)jc * ldb + l];
            } else if (opA == 'T' && opB == 'N') {
                for (int l = 0; l < k; ++l)
                    sum += A[(size_t)ic * lda + l] * B[(size_t)jc * ldb + l];
            } else if (opA == 'N' && opB == 'T') {
                for (int l = 0; l < k; ++l)
                    sum += A[(size_t)l * lda + ic] * B[(size_t)l * ldb + jc];
            } else { // 'T','T'
                for (int l = 0; l < k; ++l)
                    sum += A[(size_t)ic * lda + l] * B[(size_t)l * ldb + jc];
            }
            double cval = C[(size_t)jc * ldc + ic];
            C[(size_t)jc * ldc + ic] = alpha * sum + beta * cval;
        }
    }
}

/* Jacobi eigensolver */
static void jacobi_eigen_small(double *G, double *R, int k, int max_iter, double tol) {
    for (int j=0;j<k;++j)
        for (int i=0;i<k;++i)
            R[(size_t)j * k + i] = (i==j) ? 1.0 : 0.0;

    for (int iter=0; iter<max_iter; ++iter) {
        double max_off = 0.0; int p=-1, q=-1;
        for (int col=0; col<k; ++col)
            for (int row=0; row<col; ++row) {
                double a = fabs(G[(size_t)col * k + row]);
                if (a > max_off) { max_off = a; p = row; q = col; }
            }
        if (max_off < tol) break;
        double App = G[(size_t)p * k + p];
        double Aqq = G[(size_t)q * k + q];
        double Apq = G[(size_t)q * k + p];
        if (fabs(Apq) < 1e-18) continue;
        double tau = (Aqq - App) / (2.0 * Apq);
        double t = (tau >= 0.0 ? 1.0 : -1.0) / (fabs(tau) + sqrt(1.0 + tau*tau));
        double c = 1.0 / sqrt(1.0 + t*t);
        double s = t * c;
        for (int r = 0; r < k; ++r) {
            if (r == p || r == q) continue;
            double Grp = G[(size_t)p * k + r];
            double Grq = G[(size_t)q * k + r];
            double new_rp = c * Grp - s * Grq;
            double new_rq = s * Grp + c * Grq;
            G[(size_t)p * k + r] = new_rp;
            G[(size_t)r * k + p] = new_rp;
            G[(size_t)q * k + r] = new_rq;
            G[(size_t)r * k + q] = new_rq;
        }
        double new_pp = c*c*App - 2.0*s*c*Apq + s*s*Aqq;
        double new_qq = s*s*App + 2.0*s*c*Apq + c*c*Aqq;
        G[(size_t)p * k + p] = new_pp;
        G[(size_t)q * k + q] = new_qq;
        G[(size_t)q * k + p] = 0.0;
        G[(size_t)p * k + q] = 0.0;
        for (int r = 0; r < k; ++r) {
            double Rip = R[(size_t)p * k + r];
            double Riq = R[(size_t)q * k + r];
            R[(size_t)p * k + r] = c * Rip - s * Riq;
            R[(size_t)q * k + r] = s * Rip + c * Riq;
        }
    }
}

static void normalize_columns(double *A, int m, int n) {
    for (int col=0; col<n; ++col) {
        double s = 0.0;
        double *colptr = A + (size_t)col * m;
        for (int i=0;i<m;++i) { double v = colptr[i]; s += v*v; }
        double nrm = sqrt(s);
        if (nrm > 1e-14) {
            double inv = 1.0 / nrm;
            for (int i=0;i<m;++i) colptr[i] *= inv;
        }
    }
}

int main(void) {
    int m = 2000, n = 2000, k = 20, sweeps = 5;
    if (n % k != 0) { fprintf(stderr, "n must be divisible by k\n"); return 1; }
    printf("BCV-Jacobi WITH V (no BLAS, serial) on %dx%d, k=%d, sweeps=%d\n", m, n, k, sweeps);

    size_t m_n = (size_t)m * (size_t)n;
    size_t n_n = (size_t)n * (size_t)n;
    size_t two_k = (size_t)2 * (size_t)k;
    double *A = aligned_alloc_d(m_n);
    double *V = aligned_alloc_d(n_n);
    double *Ubuf = aligned_alloc_d((size_t)m * two_k);
    double *G = aligned_alloc_d(two_k * two_k);
    double *R = aligned_alloc_d(two_k * two_k);
    double *Utmp = aligned_alloc_d((size_t)m * two_k);
    double *Vsub = aligned_alloc_d((size_t)n * two_k);
    double *Vtmp = aligned_alloc_d((size_t)n * two_k);

    if (!A || !V || !Ubuf || !G || !R || !Utmp || !Vsub || !Vtmp) {
        fprintf(stderr,"allocation failed\n");
        return 1;
    }

    init_A(A, m, n);
    init_V_identity(V, n);

    int blocks = n / k;
    double t0 = wall_time();

    for (int sweep = 0; sweep < sweeps; ++sweep) {
        for (int q = 0; q < blocks - 1; ++q) {
            for (int jj = 0; jj < k; ++jj) {
                double *dst = Ubuf + (size_t)jj * m;
                double *src = A + (size_t)(q*k + jj) * m;
                for (int i=0;i<m;++i) dst[i] = src[i];
            }
            for (int p = q + 1; p < blocks; ++p) {
                for (int jj=0;jj<k;++jj) {
                    double *dst = Ubuf + (size_t)(k + jj) * m;
                    double *src = A + (size_t)(p*k + jj) * m;
                    for (int i=0;i<m;++i) dst[i] = src[i];
                }

                int tk = 2*k;
                dgemm_simple('T','N', tk, tk, m, 1.0, Ubuf, m, Ubuf, m, 0.0, G, tk);
                jacobi_eigen_small(G, R, tk, 200, 1e-12);
                dgemm_simple('N','N', m, tk, tk, 1.0, Ubuf, m, R, tk, 0.0, Utmp, m);
                for (int col=0; col<tk; ++col) {
                    double *src = Utmp + (size_t)col * m;
                    double *dst = Ubuf + (size_t)col * m;
                    for (int i=0;i<m;++i) dst[i] = src[i];
                }
                for (int jj=0;jj<k;++jj) {
                    double *src = V + (size_t)(q*k + jj) * n;
                    double *dst = Vsub + (size_t)jj * n;
                    for (int i=0;i<n;++i) dst[i] = src[i];
                }
                for (int jj=0;jj<k;++jj) {
                    double *src = V + (size_t)(p*k + jj) * n;
                    double *dst = Vsub + (size_t)(k + jj) * n;
                    for (int i=0;i<n;++i) dst[i] = src[i];
                }
                dgemm_simple('N','N', n, tk, tk, 1.0, Vsub, n, R, tk, 0.0, Vtmp, n);
                for (int jj=0;jj<k;++jj) {
                    double *src = Vtmp + (size_t)jj * n;
                    double *dst = V + (size_t)(q*k + jj) * n;
                    for (int i=0;i<n;++i) dst[i] = src[i];
                }
                for (int jj=0;jj<k;++jj) {
                    double *src = Vtmp + (size_t)(k + jj) * n;
                    double *dst = V + (size_t)(p*k + jj) * n;
                    for (int i=0;i<n;++i) dst[i] = src[i];
                }
                for (int jj=0;jj<k;++jj) {
                    double *src = Ubuf + (size_t)(k + jj) * m;
                    double *dst = A + (size_t)(p*k + jj) * m;
                    for (int i=0;i<m;++i) dst[i] = src[i];
                }
            }
            for (int jj=0;jj<k;++jj) {
                double *src = Ubuf + (size_t)jj * m;
                double *dst = A + (size_t)(q*k + jj) * m;
                for (int i=0;i<m;++i) dst[i] = src[i];
            }
        }
        normalize_columns(A, m, n);
    }

    double t1 = wall_time();
    printf("Elapsed time(BCV with V, serial) = %.6f seconds\n", t1 - t0);

    free(A); free(V); free(Ubuf); free(G); free(R); free(Utmp); free(Vsub); free(Vtmp);
    return 0;
}