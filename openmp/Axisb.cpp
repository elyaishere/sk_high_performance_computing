// Gauss-Seidel method
#include <cassert>
#include <chrono>
#include <cmath>
#include <ctime>
#include <functional>
#include <iostream>
#include <omp.h>
#include <random>
#include <thread>

using std::cin, std::cout, std::endl;

static const auto THREADS = std::thread::hardware_concurrency();

bool is_diagonally_dominant(double *matrix, size_t N) {
  int i, j;
  double sum;
  bool flag = 1;

#pragma omp parallel for num_threads(THREADS)
  for (i = 0; i < N; i++) {
    sum = 0;
#pragma omp parallel for reduction(+ : sum) shared(flag)
    for (j = 0; j < N; j++) {
      if (!flag)
        continue;
      sum += abs(matrix[i * N + j]);
    }
    if (sum >= 2 * abs(matrix[i * N + i])) {
      flag = 0;
    }
  }

  return flag;
}

void PrintData(double *A, double *b, size_t N) {
  cout << "Matrix A:" << endl;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j)
      cout << A[i * N + j] << " ";
    cout << endl;
  }

  cout << "vector b:" << endl;
  for (int j = 0; j < N; ++j)
    cout << b[j] << " ";
  cout << endl;
}

void FillData(double *A, double *b, double *true_x, double &eps, size_t N) {
  std::mt19937 gen;
  std::random_device rand_device;
  gen.seed(rand_device());
  std::uniform_real_distribution<double> dist(-50.0, 50.0);
  for (int i = 0; i < N; ++i) {
    true_x[i] = i; // easy to check
  }
  for (int i = 0; i < N; ++i) {
    double sum = 0;
    for (int j = 0; j < N; ++j) {
      A[i * N + j] = dist(gen);
      sum += 2 * abs(A[i * N + j]); // for more probable convergence
    }
    A[i * N + i] += sum;
  }
  for (int i = 0; i < N; ++i) {
    b[i] = 0;
    for (int j = 0; j < N; ++j)
      b[i] += A[i * N + j] * true_x[j];
  }

  while (true) {
    cout << "Enter precision: ";
    cin >> eps;

    if (eps > 0) {
      break;
    }

    cout << "Precision must be positive" << endl;
  }
}

int Solver(double *A, double *b, double *x, double *x_prev, size_t N,
           std::function<bool(void)> predicate) {
  int cnt_iter = 0;

  do {

#pragma omp parallel for num_threads(THREADS)
    for (int i = 0; i < N; ++i) {
      x_prev[i] = x[i];
    }

#pragma omp parallel for num_threads(THREADS)
    for (int i = 0; i < N; ++i) {
      double var = 0;
      for (int j = 0; j < N; ++j) {
        if (j != i) {
          var += (A[i * N + j] * x[j]);
        }
      }
      x[i] = (b[i] - var) / A[i * N + i];
    }
    ++cnt_iter;
  } while (predicate());
  return cnt_iter;
}

int main() {
  int N;
  cout << "Enter vector size: ";
  cin >> N;
  double *A, *b, *x, *x_prev, *true_x;

  A = (double *)malloc(N * N * sizeof(double));
  b = (double *)malloc(N * sizeof(double));
  x = (double *)malloc(N * sizeof(double));
  x_prev = (double *)malloc(N * sizeof(double));
  true_x = (double *)malloc(N * sizeof(double));
  double eps;

  FillData(A, b, true_x, eps, N);

  // PrintData(A, b, N);

  int iterations;

  double start;

  if (is_diagonally_dominant(A, N)) {
    start = omp_get_wtime();
    iterations = Solver(A, b, x, x_prev, N, [&x, &x_prev, N, eps]() {
      double norm = 0;
#pragma omp parallel for num_threads(THREADS) reduction(+ : norm)
      for (int i = 0; i < N; ++i) {
        norm += (x[i] - x_prev[i]) * (x[i] - x_prev[i]);
      }
      return norm > eps * eps;
    });

  } else {
    int iter_num;
    cout << "Matrix A is not diagonally dominant might not converge" << endl;
    cout << "Enter max number of iterations: ";
    cin >> iter_num;
    start = omp_get_wtime();
    iterations =
        Solver(A, b, x, x_prev, N, [&iter_num]() { return --iter_num > 0; });
  }

  double end = omp_get_wtime();

  double mse = 0;
  for (int i = 0; i < N; ++i) {
    mse += (x[i] - true_x[i]) * (x[i] - true_x[i]);
  }
  mse /= N;

  cout << "After " << iterations << " iterations and " << (end - start)
       << " seconds got solution with MSE = " << mse << endl;

  free(A);
  free(b);
  free(x);
  free(x_prev);
  free(true_x);
  return 0;
}
