// bcv_svd_timed.c
// BCV Jacobi WITH V accumulation (full SVD).
// Compile: gcc -O3 -march=native -std=c11 bcv_svd_timed.c -o bcv_svd_timed -lm
// Run example: ./bcv_svd_timed 1500 1500 30 4

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
    for (int i = 0; i < rows; ++i) M[i] = malloc(cols * sizeof(double));
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
    for (int i = 0; i < m; ++i) s += A[i][c1]*A[i][c2];
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

void apply_rotation_to_matrix(double **M, int rows, int gi, int gj, double c, double s) {
    for (int r = 0; r < rows; ++r) {
        double mi = M[r][gi];
        double mj = M[r][gj];
        M[r][gi] = c*mi - s*mj;
        M[r][gj] = s*mi + c*mj;
    }
}

void givens_rotation_local_with_V(double **U, int m, int k_local, int *col_idx,
                                  double **V, int n) {
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

            int gi = col_idx[i];
            int gj = col_idx[j];
            apply_rotation_to_matrix(V, n, gi, gj, c, s);
        }
    }
}

int main(int argc, char **argv) {
    int m = 1500, n = 1500, k = 30, sweeps = 4;
    if (argc >= 5) {
        m = atoi(argv[1]); n = atoi(argv[2]); k = atoi(argv[3]); sweeps = atoi(argv[4]);
    } else {
        printf("Usage: %s m n k sweeps  (defaults 1500 1500 30 4)\n", argv[0]);
    }
    if (n % k != 0) { fprintf(stderr, "n must be divisible by k\n"); return 1; }

    printf("BCV-Jacobi WITH V (full SVD) on %dx%d, k=%d, sweeps=%d\n", m, n, k, sweeps);

    double **A = allocate_matrix(m, n);
    double **V = allocate_matrix(n, n);

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            A[i][j] = sin((double)(i+1))*cos((double)(j+1)) + ((i+j)%11)*0.01;

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            V[i][j] = (i==j)?1.0:0.0;

    double **Ubuf = allocate_matrix(m, 2*k);
    int *col_idx = malloc(2*k * sizeof(int));

    double t0 = wall_time();

    int blocks = n/k;
    for (int sweep = 0; sweep < sweeps; ++sweep) {
        for (int q = 0; q < blocks - 1; ++q) {
            load_submatrix(Ubuf, A, m, q*k, k);
            for (int t = 0; t < k; ++t) col_idx[t] = q*k + t;

            for (int p = q + 1; p < blocks; ++p) {
                load_submatrix(Ubuf, A, m, p*k, k);
                for (int t = 0; t < k; ++t) col_idx[k+t] = p*k + t;

                for (int r = 0; r < 2*k - 1; ++r)
                    givens_rotation_local_with_V(Ubuf, m, 2*k, col_idx, V, n);

                store_submatrix(A, Ubuf, m, p*k, k);
            }
            store_submatrix(A, Ubuf, m, q*k, k);
        }
        for (int col = 0; col < n; ++col) {
            double norm = sqrt(column_norm_sq(A, m, col));
            if (norm > 1e-14) scale_column(A, m, col, 1.0/norm);
        }
    }

    double t1 = wall_time();
    printf("Elapsed time (BCV with V) = %.6f seconds\n", t1 - t0);

    free(col_idx);
    free_matrix(Ubuf, m);
    free_matrix(V, n);
    free_matrix(A, m);
    return 0;
}
