#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double pi_single_thread(const size_t N) {
  srand(time(NULL));
  double d, x, y, pi;
  int in_quadrant = 0;

  for (size_t i = 0; i < N; ++i) {
    x = ((double)rand()) / RAND_MAX;
    y = ((double)rand()) / RAND_MAX;

    d = x * x + y * y;

    in_quadrant += (d <= 1);
  }

  pi = 4.0 * in_quadrant / N;
  return pi;
}

double pi_omp(const size_t N) {
  double pi;
  int in_quadrant = 0;

#pragma omp parallel reduction(+ : in_quadrant) num_threads(10)
  {
    int id = omp_get_thread_num();
    double x, y, d;
    size_t i;

#pragma omp for private(x, y, d, id, i)
    for (i = 0; i < N; ++i) {
      x = ((double)rand_r(&id)) / RAND_MAX;
      y = ((double)rand_r(&id)) / RAND_MAX;

      d = x * x + y * y;

      in_quadrant += (d <= 1);
    }
  }

  pi = 4.0 * in_quadrant / N;
  return pi;
}

int main() {
  const size_t N = 1000000;
  double step;

  double pi;

  double start, end;

  start = omp_get_wtime();
  pi = pi_single_thread(N);
  end = omp_get_wtime();

  printf("Time elapsed (single thread): %f seconds.\n", end - start);
  printf("pi = %.16f\n", pi);

  start = omp_get_wtime();
  pi = pi_omp(N);
  end = omp_get_wtime();

  printf("Time elapsed (omp): %f seconds.\n", end - start);

  printf("pi = %.16f\n", pi);

  return 0;
}
