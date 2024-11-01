#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Constants for grid size
int M = 80, N = 90;
double h1, h2, h;

int isInTriangle(double x, double y) {
    if (y < 0 || y > 2) return 0;
    if (y == 0 && x >= -3 && x <= 3) return 1;
    double slope_AB = -2.0 / 3.0; // 斜率
    double intercept_AB = 2; // y 截距
    double y_AB = slope_AB * (x - 3); // 注意这里计算时使用 x - A_x

    double slope_BC = 2.0 / 3.0; // 斜率
    double intercept_BC = 2; // y 截距
    double y_BC = slope_BC * (x + 3);
    return (y <= y_AB && y >= 0) && (y <= y_BC && y >= 0);
}

// Function to calculate k(x,y)
double k_func(double x, double y, double h) {
    if (isInTrapezoid(x, y)) return 1.0;
    return 1.0 / (h * h); // ε = h²
}

// Function to calculate F(x,y)
double F_func(double x, double y) {
    if (isInTrapezoid(x, y)) return 1.0;
    return 0.0;
}

// Function to calculate the index in the linear system
int get_index(int i, int j, int M) {
    return (j - 1) * (M - 1) + (i - 1);
}

// Calculate Euclidean norm of difference between two vectors
double diff_norm(const double* v1, const double* v2, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Matrix-vector multiplication: y = Ax
void matrix_vector_mult(const double* A, const double* x, double* y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = 0.0;
        for (int j = 0; j < n; j++) {
            y[i] += A[i * n + j] * x[j];
        }
    }
}

// Vector dot product
double dot_product(const double* x, const double* y, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}

int main() {
    omp_set_num_threads(4); // 设置线程数量为4  // 修改的位置

    double start_time = omp_get_wtime(); // Start timing

    // Grid parameters
    double A1 = -3.0, B1 = 3.0;
    double A2 = -1.0, B2 = 2.0;
    double h1 = (B1 - A1) / M;
    double h2 = (B2 - A2) / N;
    double h = fmax(h1, h2);

    // Number of interior points
    int n_int = (M - 1) * (N - 1);

    // Allocate memory
    double* A = (double*)calloc(n_int * n_int, sizeof(double));
    double* B = (double*)calloc(n_int, sizeof(double));
    double* w = (double*)calloc(n_int, sizeof(double));
    double* w_prev = (double*)calloc(n_int, sizeof(double));
    double* r = (double*)calloc(n_int, sizeof(double));

    // Construct matrix A and vector B
#pragma omp parallel for collapse(2) // Parallelizing nested loops
    for (int j = 1; j < N; j++) {
        for (int i = 1; i < M; i++) {
            double x = A1 + i * h1;
            double y = A2 + j * h2;
            int idx = get_index(i, j, M);

            // Calculate coefficients at half points
            double a_i_minus_half = k_func(x - 0.5 * h1, y, h);
            double a_i_plus_half = k_func(x + 0.5 * h1, y, h);
            double b_j_minus_half = k_func(x, y - 0.5 * h2, h);
            double b_j_plus_half = k_func(x, y + 0.5 * h2, h);

            // Diagonal term
#pragma omp critical // Critical section to avoid race conditions
            {
                A[idx * n_int + idx] = (a_i_minus_half + a_i_plus_half) / (h1 * h1) +
                    (b_j_minus_half + b_j_plus_half) / (h2 * h2);
            }

            // Off-diagonal terms
            if (i > 1) {
                int idx_left = get_index(i - 1, j, M);
#pragma omp critical
                {
                    A[idx * n_int + idx_left] = -a_i_minus_half / (h1 * h1);
                    A[idx_left * n_int + idx] = -a_i_minus_half / (h1 * h1);
                }
            }

            if (i < M - 1) {
                int idx_right = get_index(i + 1, j, M);
#pragma omp critical
                {
                    A[idx * n_int + idx_right] = -a_i_plus_half / (h1 * h1);
                    A[idx_right * n_int + idx] = -a_i_plus_half / (h1 * h1);
                }
            }

            if (j > 1) {
                int idx_down = get_index(i, j - 1, M);
#pragma omp critical
                {
                    A[idx * n_int + idx_down] = -b_j_minus_half / (h2 * h2);
                    A[idx_down * n_int + idx] = -b_j_minus_half / (h2 * h2);
                }
            }

            if (j < N - 1) {
                int idx_up = get_index(i, j + 1, M);
#pragma omp critical
                {
                    A[idx * n_int + idx_up] = -b_j_plus_half / (h2 * h2);
                    A[idx_up * n_int + idx] = -b_j_plus_half / (h2 * h2);
                }
            }

            // Right-hand side
#pragma omp critical
            {
                B[idx] = F_func(x, y);
            }
        }
    }

    // Gradient descent parameters
    double delta = 0.00005;  // Stopping criterion threshold
    int max_iter = 1000000;

    // Gradient descent iteration
    int iter;
    double diff = INFINITY;

    for (iter = 0; iter < max_iter && diff >= delta; iter++) {
        // Save previous solution
#pragma omp parallel for
        for (int i = 0; i < n_int; i++) {
            w_prev[i] = w[i];
        }

        // Calculate residual r = B - Aw
        matrix_vector_mult(A, w, r, n_int);
#pragma omp parallel for
        for (int i = 0; i < n_int; i++) {
            r[i] = B[i] - r[i];
        }

        // Calculate τ = (r,r)/(Ar,r)
        double* Ar = (double*)calloc(n_int, sizeof(double));
        matrix_vector_mult(A, r, Ar, n_int);

        double tau = dot_product(r, r, n_int) / dot_product(Ar, r, n_int);

        // Update solution: w^(k+1) = w^(k) - τr^(k)
#pragma omp parallel for
        for (int i = 0; i < n_int; i++) {
            w[i] = w[i] + tau * r[i];
        }

        // Calculate difference norm ||w^(k+1) - w^(k)||
        diff = diff_norm(w, w_prev, n_int);

        free(Ar);

        if (iter % 100 == 0) {
            printf("Iteration %d: diff = %e\n", iter, diff);
        }
    }

    printf("Method converged in %d iterations\n", iter);
    printf("Final difference between iterations: %e\n", diff);

    // End timing
    double end_time = omp_get_wtime(); // End timing
    printf("Total execution time: %f seconds\n", end_time - start_time);

    // Save solution to file
    FILE* fp = fopen("solution.txt", "w");
    // Write boundary points (y = A2)
    for (int i = 0; i < M; i++) {
        fprintf(fp, "%f %f %f\n", A1 + i * h1, A2, 0.0);
    }
    // Write interior points
    for (int j = 1; j < N; j++) {
        fprintf(fp, "%f %f %f\n", A1, A2 + j * h2, 0.0);  // Left boundary
        for (int i = 1; i < M - 1; i++) {
            int idx = get_index(i, j, M);
            fprintf(fp, "%f %f %f\n", A1 + i * h1, A2 + j * h2, w[idx]);
        }
        fprintf(fp, "%f %f %f\n", B1, A2 + j * h2, 0.0);  // Right boundary
    }
    // Write boundary points (y = B2)
    for (int i = 0; i < M; i++) {
        fprintf(fp, "%f %f %f\n", A1 + i * h1, B2, 0.0);
    }
    fclose(fp);

    // Free memory
    free(A);
    free(B);
    free(w);
    free(w_prev);
    free(r);

    return 0;
}
