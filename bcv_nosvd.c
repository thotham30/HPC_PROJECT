// bcv_only_timed.c
// BCV Jacobi (blocked) - timed, WITHOUT accumulating V (no full SVD).
// Compile: gcc -O3 -march=native -std=c11 bcv_only_timed.c -o bcv_only_timed -lm
// Run example: ./bcv_only_timed 2000 2000 20 5

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

double wall_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

double **allocate_matrix(int rows, int cols) {
    double **M = malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; ++i) {
        M[i] = malloc(cols * sizeof(double));
    }
    return M;
}

void free_matrix(double **M, int rows) {
    for (int i = 0; i < rows; ++i) free(M[i]);
    free(M);
}

double column_norm_sq(double **A, int m, int col) {
    double s = 0.0;
    for (int i = 0; i < m; ++i) { double v = A[i][col]; s += v*v; }
    return s;
}

double column_dot(double **A, int m, int c1, int c2) {
    double s = 0.0;
    for (int i = 0; i < m; ++i) s += A[i][c1] * A[i][c2];
    return s;
}

void scale_column(double **A, int m, int col, double scale) {
    for (int i = 0; i < m; ++i) A[i][col] *= scale;
}

void load_submatrix(double **U, double **A, int m, int start_col, int k) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < k; ++j)
            U[i][j] = A[i][start_col + j];
}

void store_submatrix(double **A, double **U, int m, int start_col, int k) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < k; ++j)
            A[i][start_col + j] = U[i][j];
}

void givens_rotation_local(double **U, int m, int k_local) {
    for (int i = 0; i < k_local - 1; ++i) {
        for (int j = i + 1; j < k_local; ++j) {
            double alpha = column_norm_sq(U, m, i);
            double beta  = column_norm_sq(U, m, j);
            double gamma = column_dot(U, m, i, j);
            if (fabs(gamma) < 1e-14) continue;

            double tau = (beta - alpha) / (2.0 * gamma);
            double t = (tau >= 0.0 ? 1.0 : -1.0) / (fabs(tau) + sqrt(1.0 + tau*tau));
            double c = 1.0 / sqrt(1.0 + t*t);
            double s = t * c;

            for (int r = 0; r < m; ++r) {
                double ui = U[r][i];
                double uj = U[r][j];
                U[r][i] = c*ui - s*uj;
                U[r][j] = s*ui + c*uj;
            }
        }
    }
}

int main(int argc, char **argv) {
    int m = 2000, n = 2000, k = 20, sweeps = 5;
    if (argc >= 5) {
        m = atoi(argv[1]); n = atoi(argv[2]); k = atoi(argv[3]); sweeps = atoi(argv[4]);
    } else {
        printf("Usage: %s m n k sweeps  (defaults 2000 2000 20 5)\n", argv[0]);
    }
    if (n % k != 0) { fprintf(stderr, "n must be divisible by k\n"); return 1; }

    printf("BCV-Jacobi (no V) on %dx%d, k=%d, sweeps=%d\n", m, n, k, sweeps);

    double **A = allocate_matrix(m, n);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            A[i][j] = sin((double)(i+1))*cos((double)(j+1)) + ((i+j)%7)*0.01;

    double **U = allocate_matrix(m, 2*k);

    double t0 = wall_time();

    int blocks = n / k;
    for (int sweep = 0; sweep < sweeps; ++sweep) {
        for (int q = 0; q < blocks - 1; ++q) {
            load_submatrix(U, A, m, q*k, k);
            for (int p = q + 1; p < blocks; ++p) {
                load_submatrix(U, A, m, p*k, k);
                for (int r = 0; r < 2*k - 1; ++r) {
                    givens_rotation_local(U, m, 2*k);
                }
                store_submatrix(A, U, m, p*k, k);
            }
            store_submatrix(A, U, m, q*k, k);
        }
        for (int col = 0; col < n; ++col) {
            double norm = sqrt(column_norm_sq(A, m, col));
            if (norm > 1e-14) scale_column(A, m, col, 1.0/norm);
        }
    }

    double t1 = wall_time();
    printf("Elapsed time (BCV only) = %.6f seconds\n", t1 - t0);

    free_matrix(U, m);
    free_matrix(A, m);
    return 0;
}
