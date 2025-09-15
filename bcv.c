#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// =========================
// Utility functions
// =========================
double **allocate_matrix(int m, int n) {
    double **A = malloc(m * sizeof(double *));
    for (int i = 0; i < m; i++) {
        A[i] = malloc(n * sizeof(double));
    }
    return A;
}

void free_matrix(double **A, int m) {
    for (int i = 0; i < m; i++) free(A[i]);
    free(A);
}

double column_norm(double **A, int m, int col) {
    double sum = 0.0;
    for (int i = 0; i < m; i++) sum += A[i][col] * A[i][col];
    return sum;
}

double column_dot(double **A, int m, int col1, int col2) {
    double sum = 0.0;
    for (int i = 0; i < m; i++) sum += A[i][col1] * A[i][col2];
    return sum;
}

void scale_column(double **A, int m, int col, double scale) {
    for (int i = 0; i < m; i++) A[i][col] *= scale;
}

// =========================
// Submatrix ops
// =========================
void load_submatrix(double **U, double **A, int m, int start_col, int k) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++)
            U[i][j] = A[i][start_col + j];
}

void store_submatrix(double **A, double **U, int m, int start_col, int k) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++)
            A[i][start_col + j] = U[i][j];
}

// =========================
// Jacobi rotation
// =========================
void givens_rotation(double **U, int m, int k) {
    for (int i = 0; i < k - 1; i++) {
        for (int j = i + 1; j < k; j++) {
            double alpha = column_norm(U, m, i);
            double beta  = column_norm(U, m, j);
            double gamma = column_dot(U, m, i, j);

            if (fabs(gamma) < 1e-12) continue;

            double tau = (beta - alpha) / (2.0 * gamma);
            double t = (tau >= 0 ? 1.0 : -1.0) /
                       (fabs(tau) + sqrt(1.0 + tau * tau));
            double c = 1.0 / sqrt(1.0 + t * t);
            double s = t * c;

            for (int row = 0; row < m; row++) {
                double ui = U[row][i];
                double uj = U[row][j];
                U[row][i] = c * ui - s * uj;
                U[row][j] = s * ui + c * uj;
            }
        }
    }
}

// =========================
// BCV Jacobi (sequential)
// =========================
void bcv_jacobi(double **A, int m, int n, int k, int num_sweeps) {
    double **U = allocate_matrix(m, 2 * k);

    for (int sweep = 0; sweep < num_sweeps; sweep++) {
        for (int q = 0; q < n / k - 1; q++) {
            load_submatrix(U, A, m, q * k, k);

            for (int p = q + 1; p < n / k; p++) {
                load_submatrix(U, A, m, p * k, k);

                for (int r = 0; r < 2 * k - 1; r++) {
                    givens_rotation(U, m, 2 * k);
                }

                store_submatrix(A, U, m, p * k, k);
            }

            store_submatrix(A, U, m, q * k, k);
        }

        // normalize columns (singular value approx)
        for (int i = 0; i < n; i++) {
            double norm = sqrt(column_norm(A, m, i));
            if (norm > 1e-12) scale_column(A, m, i, 1.0 / norm);
        }
    }

    free_matrix(U, m);
}

// =========================
// Test driver
// =========================
int main() {
    int m = 500, n = 500, k = 10, sweeps = 5;
    double **A = allocate_matrix(m, n);

    // Initialize matrix with some values (example: A[i][j] = i + j)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = (i + j) % 10 + 1;  // simple pattern, not all zeros
        }
    }

    printf("Running BCV Jacobi on %dx%d matrix...\n", m, n);

    bcv_jacobi(A, m, n, k, sweeps);

    printf("Finished.\n");

    // Optionally print just first 5x5 block to verify
    printf("\nTop-left 5x5 block of processed matrix:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%8.4f ", A[i][j]);
        }
        printf("\n");
    }

    free_matrix(A, m);
    return 0;
}

