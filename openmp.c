#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WIDTH 160
#define HEIGHT 180
#define MC_ITER 1000 // Monte Carlo iterations
#define DELTA 1e-5  // Stop criterion
#define MIN_X -4.0
#define MAX_X 4.0
#define MIN_Y -1.0
#define MAX_Y 3.0

#define IDX(row, col) (row) * (WIDTH + 1) + (col)

double x_step = (MAX_X - MIN_X) / WIDTH;  // Grid size X
double y_step = (MAX_Y - MIN_Y) / HEIGHT; // Grid size Y
double eps;

int size = (WIDTH + 1) * (HEIGHT + 1);

// Structures
typedef struct {
  double x, y;
} Point;

typedef struct {
  Point top_left;
  Point bottom_right;
} Rect;

// Global points defining area D
const Point A = {-3.0, 0.0};
const Point B = {3.0, 0.0};
const Point C = {0.0, 2.0};

// Global rectangle R
const Rect R = {{MIN_X, MAX_Y}, {MAX_X, MIN_Y}};

// Utility functions
void log_time(const char *msg) {
  time_t now;
  time(&now);
  printf("[%.24s] %s\n", ctime(&now), msg);
}

void PrintMatr_tofile(FILE *fout, double *matrix) {
  for (int row = 0; row < HEIGHT + 1; row++) {
    for (int col = 0; col < WIDTH + 1; col++) {
      fprintf(fout, "%f ", matrix[IDX(row, col)]);
    }
    fprintf(fout, "\n");
  }
}

// Check if point is in area D
int in_area_d(Point p) {
  // Trapezoid vertices: A(-3,0), B(3,0), C(0,2)
  if (p.y < 0 || p.y > 2)
    return 0;
  if (p.y == 0 && p.x >= -3 && p.x <= 3)
    return 1;
  double x_left = -3 + 3 * p.y / 2;
  double x_right = 3 - 3 * p.y / 2;
  return (p.x >= x_left && p.x <= x_right);
}
double func_k(double x, double y) {
  Point p = {x, y};
  if (in_area_d(p))
    return 1.0;
  double h = fmax(x_step, y_step);
  return 1.0 / (h * h); // ε = h²
}

double func_F(double x, double y) {
  Point p = {x, y};
  if (in_area_d(p))
    return 1.0;
  return 0.0;
}

// Vector operations
double vec_dot(const double *v1, const double *v2, int size) {
  double sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < size; i++) {
    sum += v1[i] * v2[i];
  }
  return sum;
}

double vec_norm(const double *v, int size) { return sqrt(vec_dot(v, v, size)); }

void vec_sub(double *result, const double *v1, const double *v2, int size) {
#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    result[i] = v1[i] - v2[i];
  }
}

void vec_scale(double *result, const double *v, double scale, int size) {
#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    result[i] = v[i] * scale;
  }
}

// Calculate coefficient a_ij
double calc_a_coef(int row, int col) {
  double y = MIN_Y + y_step * row;
  double x = MIN_X + x_step * col;
  double xm12 = x - x_step * 0.5;
  double ym12 = y - y_step * 0.5;
  double yp12 = y + y_step * 0.5;

  int p1_in_d = in_area_d((Point){xm12, ym12});
  int p2_in_d = in_area_d((Point){xm12, yp12});
  if (p1_in_d && p2_in_d)
    return 1.0;
  else if (!p1_in_d && !p2_in_d)
    return 1.0 / eps;

  int count = 0;
#pragma omp parallel for reduction(+ : count)
  for (int k = 0; k < MC_ITER; k++) {
    double rand_y = ym12 + y_step * (double)rand() / (double)RAND_MAX;
    if (in_area_d((Point){xm12, rand_y}))
      count++;
  }

  double l = (double)count / (double)MC_ITER;
  return l + (1.0 - l) / eps;
}

// Calculate coefficient b_ij
double calc_b_coef(int row, int col) {
  double y = MIN_Y + y_step * row;
  double x = MIN_X + x_step * col;
  double xm12 = x - x_step * 0.5;
  double xp12 = x + x_step * 0.5;
  double ym12 = y - y_step * 0.5;

  int p1_in_d = in_area_d((Point){xm12, ym12});
  int p2_in_d = in_area_d((Point){xp12, ym12});
  if (p1_in_d && p2_in_d)
    return 1.0;
  else if (!p1_in_d && !p2_in_d)
    return 1.0 / eps;

  int count = 0;
#pragma omp parallel for reduction(+ : count)
  for (int k = 0; k < MC_ITER; k++) {
    double rand_x = xm12 + x_step * (double)rand() / (double)RAND_MAX;
    if (in_area_d((Point){rand_x, ym12}))
      count++;
  }

  double l = (double)count / (double)MC_ITER;
  return l + (1.0 - l) / eps;
}

// Calculate fictitious area square
double calc_F_coef(int row, int col) {
  double y = MIN_Y + y_step * row;
  double x = MIN_X + x_step * col;
  double xm12 = x - x_step * 0.5;
  double xp12 = x + x_step * 0.5;
  double ym12 = y - y_step * 0.5;
  double yp12 = y + y_step * 0.5;

  int p1 = in_area_d((Point){xm12, ym12});
  int p2 = in_area_d((Point){xp12, ym12});
  int p3 = in_area_d((Point){xp12, yp12});
  int p4 = in_area_d((Point){xm12, yp12});

  if (p1 && p2 && p3 && p4)
    return 1.0;
  else if (!p1 && !p2 && !p3 && !p4)
    return 0.0;

  int count = 0;
#pragma omp parallel for reduction(+ : count)
  for (int i = 0; i < MC_ITER; i++) {
    double x = xm12 + x_step * (double)rand() / (double)RAND_MAX;
    double y = ym12 + y_step * (double)rand() / (double)RAND_MAX;
    if (in_area_d((Point){x, y}))
      count++;
  }

  return (double)count / (double)MC_ITER;
}

// Aw by formula (10)
void mat_vec_mul(double *res, const double *a, const double *b, const double *w,
                 int size) {
#pragma omp parallel for collapse(2)
  for (int row = 0; row < HEIGHT + 1; row++) {
    for (int col = 0; col < WIDTH + 1; col++) {
      if (row == 0 || row == HEIGHT || col == 0 || col == WIDTH) {
        res[IDX(row, col)] = w[IDX(row, col)];
      } else {
        double t1 = a[IDX(row, col + 1)] *
                    (w[IDX(row, col + 1)] - w[IDX(row, col)]) / x_step;
        double t2 = a[IDX(row, col)] *
                    (w[IDX(row, col)] - w[IDX(row, col - 1)]) / x_step;
        double t3 = b[IDX(row + 1, col)] *
                    (w[IDX(row + 1, col)] - w[IDX(row, col)]) / y_step;
        double t4 = b[IDX(row, col)] *
                    (w[IDX(row, col)] - w[IDX(row - 1, col)]) / y_step;

        res[IDX(row, col)] = -(t1 - t2) / x_step - (t3 - t4) / y_step;
      }
    }
  }
}

// Solve system of linear equations ( метод наискорейшего спуска)
double *solve_sle(const double *a, const double *b, const double *B, int size) {
  log_time("Starting SLE solver");

  double *w_k = calloc(size, sizeof(double));
  double *w_next = calloc(size, sizeof(double));
  double *r_k = calloc(size, sizeof(double));
  double *Ar_k = calloc(size, sizeof(double));
  double *temp = calloc(size, sizeof(double));

  for (int iter = 0;; iter++) {
    // Calculate residual
    mat_vec_mul(temp, a, b, w_k, size);//Aw_k
    vec_sub(r_k, temp, B, size); // r_k = Aw-B

    // Calculate iteration parameter
    mat_vec_mul(Ar_k, a, b, r_k, size); // Ar_k
    double rkk = vec_dot(r_k, r_k, size);// (rk,rk)
    double tau = rkk / vec_dot(Ar_k, r_k, size);// tau = (rk,rk)/(Ar_k,rk)

    // Calculate next iteration
    vec_scale(temp, r_k, tau, size); // rk*tau
    vec_sub(w_next, w_k, temp, size); // w_next = wk - (rk*tau)

    // Check convergence
    double err = sqrt(rkk) * tau;
    if (iter % 2000 == 0) {
      printf("Iteration %d, error: %g\n", iter, err);
    }

    if (err < DELTA) {
      printf("Converged after %d iterations, error: %g\n", iter, err);
      break;
    }

    // Swap pointers
    double *tmp = w_k;
    w_k = w_next;
    w_next = tmp;
  }

  free(w_k);
  free(r_k);
  free(Ar_k);
  free(temp);

  return w_next;
}


int main() {
  double h = fmax(x_step, y_step);
  eps = h * h;

  srand(0);
  log_time("Starting program");

  double *B = calloc(size, sizeof(double));
  double *a = calloc(size, sizeof(double));
  double *b = calloc(size, sizeof(double));

#pragma omp parallel for collapse(2)
  for (int row = 1; row < HEIGHT; row++) {
    for (int col = 1; col < WIDTH; col++) {
      B[IDX(row, col)] = calc_F_coef(row, col);
    }
  }
#pragma omp parallel for collapse(2)
  for (int row = 0; row < HEIGHT + 1; row++) {
    for (int col = 0; col < WIDTH + 1; col++) {
      a[IDX(row, col)] = calc_a_coef(row, col);
      b[IDX(row, col)] = calc_b_coef(row, col);
    }
  }
  // Solve system
  double start = omp_get_wtime();
  double *result = solve_sle(a, b, B, size);
  double end = omp_get_wtime();

  // 保存解的结果   并写入txt文件
  FILE *wout = fopen("result.txt", "w");
  PrintMatr_tofile(wout, result);
  fclose(wout);

/*   FILE *fout = fopen("F.txt", "w");
  PrintMatr_tofile(fout, B);
  fclose(fout);
  FILE *aout = fopen("a.txt", "w");
  PrintMatr_tofile(fout, a);
  fclose(aout);
  FILE *bout = fopen("b.txt", "w");
  PrintMatr_tofile(fout, b);
  fclose(bout); */

  printf("Execution time: %.2f seconds\n", end - start);

  // Cleanup
  free(B);
  free(a);
  free(b);
  free(result);

  log_time("Program finished");
  return 0;
}
