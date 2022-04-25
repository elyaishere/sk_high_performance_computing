#include <fstream>
#include <functional>
#include <memory>
#include <mpi.h>
#include <random>
#include <stdexcept>
#include <vector>

/// I understand random passing as:
///     - forbidden to pass itself
///     - allowed to pass ball towards processes which already had it in the
///     past
///     - some processes can never pass the ball
int get_random_recver(int current, int size) {
  std::random_device rd;
  std::mt19937 engine(rd());
  std::vector<int> d(size, 1);
  d[current] = 0;
  std::discrete_distribution<> dist{d.begin(), d.end()};
  auto rng = std::bind(dist, std::ref(engine));

  int r = rng();

  return r;
}

struct Ball {
public:
  /// trows if size < 0
  /// actual size will be size + 2
  Ball(int size, int number_of_passes) {
    if (size < 0) {
      throw std::runtime_error("Size less than 0");
    }
    data_.resize(size + 2);
    data_[0] = number_of_passes; // overall number of passes
    data_[1] = 0;                // current number of passes
                  // data_[2 : number_of_passes + 2] - just for measurements
  }
  size_t Size() const { return data_.size(); }
  int &operator[](size_t i) { return data_[i]; }

private:
  std::vector<int> data_;
};

struct Process {
public:
  /// throws if there are less than 2 processes
  Process(Ball &ball, bool is_game_starter = false)
      : ball_(ball), is_game_starter_(is_game_starter) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);

    if (size_ < 2) {
      throw std::runtime_error("Less than 2 processes");
    }
  }

  void Play() {
    if (is_game_starter_) {
      if (EndGameIfEnoughPasses()) {
        return;
      }
      PreparePass();
      MPI_Ssend(&ball_[0], ball_.Size(), MPI_INT, target_, 0, MPI_COMM_WORLD);
    }
    while (true) {
      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &st_);
      MPI_Recv(&ball_[0], ball_.Size(), MPI_INT, st_.MPI_SOURCE, 0,
               MPI_COMM_WORLD, &st_);
      if (EndGameIfEnoughPasses()) {
        break;
      }
      PreparePass();
      MPI_Ssend(&ball_[0], ball_.Size(), MPI_INT, target_, 0, MPI_COMM_WORLD);
    }
  }

private:
  bool EndGameIfEnoughPasses() {
    if (ball_[0] == -1) {
      return true;
    }
    if (ball_[0] == ball_[1]) {
      ball_[0] = -1;
      for (int i = 0; i < size_; ++i) {
        if (i != rank_) {
          MPI_Send(&ball_[0], ball_.Size(), MPI_INT, i, 0, MPI_COMM_WORLD);
        }
      }
      return true;
    }
    return false;
  }

  void PreparePass() {
    ball_[1]++;
    target_ = get_random_recver(rank_, size_);
  }

private:
  Ball &ball_;
  int rank_;
  int size_;
  int target_;
  bool is_game_starter_{false};
  MPI_Status st_;
};

auto InitProcess(Ball &ball, int rank) {
  if (rank == 0) {
    return std::unique_ptr<Process>(new Process(ball, true));
  } else {
    return std::unique_ptr<Process>(new Process(ball));
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int passes_number = 5, datasize = 0;
  std::ofstream file("stats.txt", std::ios_base::app);

  Ball ball(datasize, passes_number);

  auto p = InitProcess(ball, rank);

  double start = MPI_Wtime();

  p->Play();

  double end = MPI_Wtime();

  auto totalTime = end - start;
  auto timePerPass = totalTime / passes_number;
  auto ballSizeInMB = datasize * sizeof(int) / 1000000.0;

  if (rank == 0) {
    file << "Ball size (MB): " << ballSizeInMB << ";";
    file << "Total time (Sec): " << totalTime << ";";
    file << "Time per pass (Sec): " << timePerPass << ";";
    file << "Bandwidth (MB/s): " << ballSizeInMB / timePerPass << std::endl;
  }

  MPI_Finalize();
}
