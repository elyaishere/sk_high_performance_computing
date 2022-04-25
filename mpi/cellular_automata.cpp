#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <stdexcept>
#include <vector>

using std::cin, std::cout, std::endl, std::vector;

/// Initialize vector of future states according to rule (0-255)
void GenTransforms(int rule, vector<int> &transforms) {
  int ptr = 0;
  while (rule > 0) {
    transforms[ptr++] = rule % 2;
    rule /= 2;
  }
}

// Construct index in transforms array
int GetIdx(int8_t left, int8_t mid, int8_t right) {
  return right + mid * 2 + left * 4;
}

struct Process {
public:
  /// throws if there are less than 2 processes
  Process(bool is_init = false) : transforms_(8, 0), is_init_(is_init) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);

    if (size_ < 2) {
      throw std::runtime_error("Less than 2 processes");
    }

    if (is_init_) {
      do {
        cout << "Enter positive array size: ";
        cin >> N_;
      } while (N_ <= 0); // check correctness

      // any rule
      do {
        cout << "Enter rule number from 0 to 255: ";
        cin >> rule_;
      } while (rule_ < 0 || rule_ > 255); // check correctness

      do {
        cout << "Enter iterations number: ";
        cin >> iterations_number_;
      } while (iterations_number_ <= 0); // check correctness

      // Implement both periodic and constant boundary conditions
      cout << "Set periodic boundary conditions? (1/0): ";
      cin >> periodic_boundary_conditions_;

      GenTransforms(rule_, transforms_);
      buffer_.resize(N_);
      buffer_[N_ / 2] = 1;
      chunk_ = (N_ - 1) / (size_ - 1) + 1;

      printf("Initial state: ");
      for (auto i : buffer_) {
        printf("%d ", i);
      }
      printf("\n");

      int ibeg = 0;

      for (int i = 1; i < size_; ++i) {
        // Send general information to worker threads
        MPI_Ssend(&N_, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
        MPI_Ssend(&rule_, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
        MPI_Ssend(&iterations_number_, 1, MPI_INT, i, 3, MPI_COMM_WORLD);
        MPI_Ssend(&transforms_[0], transforms_.size(), MPI_INT, i, 4,
                  MPI_COMM_WORLD);

        // Validate current chunk size;
        int chunk = chunk_;
        if (ibeg + chunk > N_) {
          chunk = std::max(N_ - ibeg, 0);
        }
        chunks_.push_back(chunk);

        // Divide the computational domain into chunks and allocate them to
        // different processes
        MPI_Ssend(&chunk, 1, MPI_INT, i, 5, MPI_COMM_WORLD);
        MPI_Ssend(&buffer_[ibeg], chunk, MPI_INT8_T, i, 6, MPI_COMM_WORLD);

        ibeg += chunk;
      }

    } else {
      MPI_Recv(&N_, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &st_);
      MPI_Recv(&rule_, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &st_);
      MPI_Recv(&iterations_number_, 1, MPI_INT, 0, 3, MPI_COMM_WORLD, &st_);
      MPI_Recv(&transforms_[0], transforms_.size(), MPI_INT, 0, 4,
               MPI_COMM_WORLD, &st_);
      MPI_Recv(&chunk_, 1, MPI_INT, 0, 5, MPI_COMM_WORLD, &st_);

      buffer_.resize(chunk_);

      MPI_Recv(&buffer_[0], chunk_, MPI_INT8_T, 0, 6, MPI_COMM_WORLD, &st_);
    }
  }

  void Run() {
    while (iterations_number_-- > 0) {
      if (is_init_) {
        // Actual chunks are stored in worker threads
        // this thread sends to workers only left and right borders of
        // particular chunk
        left_ = (periodic_boundary_conditions_) ? buffer_[N_ - 1] : 0;
        right_ = (periodic_boundary_conditions_) ? buffer_[0] : 0;
        int ibeg = 0;
        for (int i = 1; i < size_; ++i) {
          // borders to send to worker i
          int left = (ibeg == 0) ? left_ : buffer_[ibeg - 1];
          int right = (ibeg + chunks_[i - 1] == N_)
                          ? right_
                          : buffer_[ibeg + chunks_[i - 1]];

          MPI_Ssend(&left, 1, MPI_INT8_T, i, 0, MPI_COMM_WORLD);
          MPI_Ssend(&right, 1, MPI_INT8_T, i, 1, MPI_COMM_WORLD);
          ibeg += chunks_[i - 1];
        }

        // now wait for new borders from workers
        ibeg = 0;
        for (int i = 1; i < size_; ++i) {
          int left, right;
          MPI_Recv(&left, 1, MPI_INT8_T, i, 2, MPI_COMM_WORLD, &st_);
          MPI_Recv(&right, 1, MPI_INT8_T, i, 3, MPI_COMM_WORLD, &st_);
          if (chunks_[i - 1] > 0) {
            buffer_[ibeg] = left;
            buffer_[ibeg + chunks_[i - 1] - 1] = right;
          }
          ibeg += chunks_[i - 1];
        }

      } else {
        // get new borders
        MPI_Recv(&left_, 1, MPI_INT8_T, 0, 0, MPI_COMM_WORLD, &st_);
        MPI_Recv(&right_, 1, MPI_INT8_T, 0, 1, MPI_COMM_WORLD, &st_);
        for (int i = 0; i < chunk_; ++i) {
          auto right = (i + 1 == chunk_) ? right_ : buffer_[i + 1];
          auto idx = GetIdx(left_, buffer_[i], right);

          left_ = buffer_[i];
          buffer_[i] = transforms_[idx];
        }

        if (chunk_ > 0) {
          left_ = buffer_[0];
          right_ = buffer_[chunk_ - 1];
        }
        // send chunk new borders
        MPI_Ssend(&left_, 1, MPI_INT8_T, 0, 2, MPI_COMM_WORLD);
        MPI_Ssend(&right_, 1, MPI_INT8_T, 0, 3, MPI_COMM_WORLD);
      }
    } // end of iterations

    if (is_init_) {
      int ibeg = 0;
      for (int i = 1; i < size_; ++i) {
        MPI_Recv(&buffer_[ibeg], chunks_[i - 1], MPI_INT8_T, i, 0,
                 MPI_COMM_WORLD, &st_);
        ibeg += chunks_[i - 1];
      }
      printf("Result: ");
      for (int i = 0; i < N_; ++i) {
        printf("%d ", buffer_[i]);
      }
      printf("\n");
    } else {
      // send all chunks to main thread
      MPI_Ssend(&buffer_[0], chunk_, MPI_INT8_T, 0, 0, MPI_COMM_WORLD);
    }
  }

private:
  vector<int8_t> buffer_;
  int8_t left_{0};
  int8_t right_{0};
  vector<int> chunks_;
  vector<int> transforms_;
  int N_;
  int rule_;
  int iterations_number_;
  int chunk_;
  int rank_;
  int size_;
  bool is_init_{false};
  MPI_Status st_;
  bool periodic_boundary_conditions_;
};

auto InitProcess(int rank) {
  if (rank == 0) {
    return std::unique_ptr<Process>(new Process(true));
  } else {
    return std::unique_ptr<Process>(new Process());
  }
}

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::ofstream file("speed.txt", std::ios_base::app);

  auto p = InitProcess(rank);

  double start = MPI_Wtime();
  p->Run();
  double end = MPI_Wtime();

  if (rank == 0) {
    file << "elapsed time: " << end - start << endl;
  }

  MPI_Finalize();
}
