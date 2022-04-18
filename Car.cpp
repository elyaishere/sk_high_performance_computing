#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <thread>

// include omp header file here
#include <omp.h>

#define RGB_COMPONENT_COLOR 255

static const auto THREADS = std::thread::hardware_concurrency();

struct PPMPixel {
  int red;
  int green;
  int blue;
};

typedef struct {
  int x, y, all;
  PPMPixel *data;
} PPMImage;

void readPPM(const char *filename, PPMImage &img) {
  std::ifstream file(filename);
  if (file) {
    std::string s;
    int rgb_comp_color;
    file >> s;
    if (s != "P3") {
      std::cout << "error in format" << std::endl;
      exit(9);
    }
    file >> img.x >> img.y;
    file >> rgb_comp_color;
    img.all = img.x * img.y;
    std::cout << s << std::endl;
    std::cout << "x=" << img.x << " y=" << img.y << " all=" << img.all
              << std::endl;
    img.data = new PPMPixel[img.all];
    for (int i = 0; i < img.all; i++) {
      file >> img.data[i].red >> img.data[i].green >> img.data[i].blue;
    }

  } else {
    std::cout << "the file:" << filename << "was not found" << std::endl;
  }
  file.close();
}

void writePPM(const char *filename, PPMImage &img) {
  std::ofstream file(filename, std::ofstream::out);
  file << "P3" << std::endl;
  file << img.x << " " << img.y << " " << std::endl;
  file << RGB_COMPONENT_COLOR << std::endl;

  for (int i = 0; i < img.all; i++) {
    file << img.data[i].red << " " << img.data[i].green << " "
         << img.data[i].blue << (((i + 1) % img.x == 0) ? "\n" : " ");
  }
  file.close();
}

// the function for shifting

void shiftPPM_omp(PPMImage &img, int shift) {
  PPMImage new_img;
  new_img.data = new PPMPixel[img.all];

  for (int k = 0; k < shift; ++k) {

#pragma omp parallel for num_threads(THREADS)
    for (int i = 0; i < img.y; i++) {
      for (int j = 0; j < img.x; j++) {
        auto &data = new_img.data[(i * img.x + j + 1) % img.all];
        auto &old_data = img.data[i * img.x + j];
        data.red = old_data.red;
        data.green = old_data.green;
        data.blue = old_data.blue;
      }
    }

#pragma omp parallel for num_threads(THREADS)
    for (int i = 0; i < img.all; i++) {
      auto &data = img.data[i];
      auto &new_data = new_img.data[i];
      data.red = new_data.red;
      data.green = new_data.green;
      data.blue = new_data.blue;
    }
  }
}

void shiftPPM(PPMImage &img, int shift) {
  PPMImage new_img;
  new_img.data = new PPMPixel[img.all];

  for (int k = 0; k < shift; ++k) {
    for (int i = 0; i < img.y; i++) {
      for (int j = 0; j < img.x; j++) {
        auto &data = new_img.data[(i * img.x + j + 1) % img.all];
        auto &old_data = img.data[i * img.x + j];
        data.red = old_data.red;
        data.green = old_data.green;
        data.blue = old_data.blue;
      }
    }

    for (int i = 0; i < img.all; i++) {
      auto &data = img.data[i];
      auto &new_data = new_img.data[i];
      data.red = new_data.red;
      data.green = new_data.green;
      data.blue = new_data.blue;
    }
  }
}

int main(int argc, char *argv[]) {
  PPMImage image;
  readPPM("car.ppm", image);

  int shift = 400;

  double start, end;

  start = omp_get_wtime();
  shiftPPM(image, shift);
  end = omp_get_wtime();
  printf("Time elapsed %d shifts: %f seconds.\n", shift, end - start);

  writePPM("new_car_1.ppm", image);
  readPPM("car.ppm", image);

  start = omp_get_wtime();
  shiftPPM_omp(image, shift);
  end = omp_get_wtime();
  printf("Time elapsed %d shifts  %d threads (omp): %f seconds.\n", shift,
         THREADS, end - start);

  writePPM("new_car_2.ppm", image);
  return 0;
}
