#include <cassert>
#include <chrono>
#include <cmath>
#include <ctime>
#include <functional>
#include <iostream>
#include <omp.h>
#include <random>
#include <thread>
#include <utility>

using std::cin, std::cout, std::endl;

static const auto THREADS = std::thread::hardware_concurrency();

void InitData(double *x, double *y, double *y_true, size_t N, double a, double b) {
  std::mt19937 gen;
  std::random_device rand_device;
  gen.seed(rand_device());
  std::uniform_real_distribution<double> dist(-50.0, 50.0);

  for (int i = 0; i < N; ++i) {
    x[i] = dist(gen);
    y[i] = 0;
    y_true[i] = a * x[i] + b;
  }
}

std::pair<double, double> Solver(double *x, double *y, size_t N, int iter_num,
                                 double learning_rate, double *y_true) {
  double a_hat = 0, b_hat = 0;

  for (int j = 0; j < iter_num; ++j) {

#pragma omp parallel for num_threads(THREADS) reduction(+: a_hat, b_hat)
    for (size_t i = 0; i < N; ++i) {
        auto diff = y[i] - y_true[i];
        a_hat -= 2 * diff * x[i] * learning_rate / N;
        b_hat -= 2 * diff * learning_rate / N;
    }

#pragma omp parallel for num_threads(THREADS)
    for (size_t i = 0; i < N; ++i) {
        y[i] = a_hat * x[i] + b_hat;
    }
  }
  return {a_hat, b_hat};
}

int main() {
  int N;
  cout << "Enter vector size: ";
  cin >> N;
  cout << "Enter a and b: ";
  double a, b;
  cin >> a >> b;
  double *x, *y, *y_true;
  x = (double *)malloc(N * sizeof(double));
  y = (double *)malloc(N * sizeof(double));
  y_true = (double *)malloc(N * sizeof(double));
  cout << "Enter GD iterations number: ";
  int iter_num;
  cin >> iter_num;
  double learning_rate;
  cout << "Enter GD step: ";
  cin >> learning_rate;

  InitData(x, y, y_true, N, a, b);

  double start = omp_get_wtime();

  auto ab = Solver(x, y, N, iter_num, learning_rate, y_true);

  double end = omp_get_wtime();

  cout << "After " << (end - start) << " seconds a_hat = " << ab.first
       << ", a = " << a << ", b_hat = " << ab.second << ", b = " << b << endl;

  free(x);
  free(y);
  free(y_true);
  return 0;
}
