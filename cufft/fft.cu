#include <chrono> 

#include "fft_cpu.hpp"

using namespace chrono;


template <typename T>
ostream &operator<<(ostream &o, vector<T> v) {
    if (v.size() > 0)
        o << v[0];
    for (size_t i = 1; i < v.size(); i++)
        o << " " << v[i];
    return o << endl;
}

static __device__ __host__ inline f2complex Add(f2complex A, f2complex B) {
    f2complex C;
    C.x = A.x + B.x;
    C.y = A.y + B.y;
    return C;
}

static __device__ __host__ inline f2complex Inverse(f2complex A) {
    f2complex C;
    C.x = -A.x;
    C.y = -A.y;
    return C;
}

static __device__ __host__ inline f2complex Multiply(f2complex A, f2complex B) {
    f2complex C;
    C.x = A.x * B.x - A.y * B.y;
    C.y = A.y * B.x + A.x * B.y;
    return C;
}

__global__ void inplace_divide_invert(f2complex *A, int n, int threads) {
    int i = blockIdx.x * threads + threadIdx.x;
    if (i < n) {
        A[i].x /= n;
        A[i].y /= n;
    }
}

__global__ void bitrev_reorder(f2complex *__restrict__ r, f2complex *__restrict__ d, int s, size_t nthr, int n) {
    int id = blockIdx.x * nthr + threadIdx.x;
    if (id < n and __brev(id) >> (32 - s) < n)
        r[__brev(id) >> (32 - s)] = d[id];
}


__device__ void inplace_fft_inner(f2complex *__restrict__ A, int i, int j, int len, int n, bool invert) {
    if (i + j + len / 2 < n and j < len / 2) {
        f2complex u, v;

        float angle = (2 * M_PI * j) / (len * (invert ? -1.0 : 1.0));
        v.x = cos(angle);
        v.y = sin(angle);

        u = A[i + j];
        v = Multiply(A[i + j + len / 2], v);
        A[i + j] = Add(u, v);
        A[i + j + len / 2] = Add(u, Inverse(v));
    }
}

__global__ void inplace_fft(f2complex *__restrict__ A, int i, int len, int n, int threads, bool invert) {
    int j = blockIdx.x * threads + threadIdx.x;
    inplace_fft_inner(A, i, j, len, n, invert);
}

__global__ void inplace_fft_outer(f2complex *__restrict__ A, int len, int n, int threads, bool invert)
{
    int i = blockIdx.x * threads + threadIdx.x;
    for (int j = 0; j < len / 2; j++) {
        inplace_fft_inner(A, i, j, len, n, invert);
    }
}

int cufft(vector<fcomplex> &a, bool invert, int balance = 10, int threads = 32) {

    int n = (int)a.size();
    int data_size = n * sizeof(f2complex);
    f2complex *data_array = (f2complex *)malloc(data_size);
    for (int i = 0; i < n; i++) {
        data_array[i].x = a[i].real();
        data_array[i].y = a[i].imag();
    }
    

    f2complex *A, *dn;
    cudaMalloc((void **)&A, data_size);
    cudaMalloc((void **)&dn, data_size);
    cudaMemcpy(dn, data_array, data_size, cudaMemcpyHostToDevice);

    auto start = high_resolution_clock::now();
    int s = log2(n);

    bitrev_reorder<<<ceil(float(n) / threads), threads>>>(A, dn, s, threads, n);

    cudaDeviceSynchronize();

    for (int len = 2; len <= n; len <<= 1) {
        if (n / len > balance) {
            inplace_fft_outer<<<ceil((float)n / threads), threads>>>(A, len, n, threads, invert);
        }
        else {
            for (int i = 0; i < n; i += len) {
                float repeats = len / 2;
                inplace_fft<<<ceil(repeats / threads), threads>>>(A, i, len, n, threads, invert);
            }
        }
    }
    
    if (invert)
        inplace_divide_invert<<<ceil(n * 1.00 / threads), threads>>>(A, n, threads);

    auto stop = high_resolution_clock::now(); 
    auto duration = duration_cast<microseconds>(stop - start); 

    f2complex *result;
    result = (f2complex *)malloc(data_size);
    cudaMemcpy(result, A, data_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++)
    {
        a[i] = fcomplex(result[i].x, result[i].y);
    }

    free(data_array);
    cudaFree(A);
    cudaFree(dn);
    return duration.count();
}

/// Function to multiply two polynomial with cuda
vector<int> cumult(vector<int> a, vector<int> b, int balance, int threads, int&d)
{
    vector<fcomplex> fa(a.begin(), a.end()), fb(b.begin(), b.end());

    // padding with zero to make their size equal to power of 2
    size_t n = 1;
    while (n < max(a.size(), b.size()))
        n <<= 1;
    n <<= 1;

    fa.resize(n), fb.resize(n);

    auto d1 = cufft(fa, false, balance, threads);
    auto d2 = cufft(fb, false, balance, threads);

    auto start = high_resolution_clock::now();
    for (size_t i = 0; i < n; ++i)
        fa[i] *= fb[i];
    
    auto stop = high_resolution_clock::now(); 
    auto duration = duration_cast<microseconds>(stop - start); 

    auto d3 = cufft(fa, true, balance, threads);

    vector<int> res;
    res.resize(n);
    for (size_t i = 0; i < n; ++i)
        res[i] = int(fa[i].real() + 0.5);

    d = d1 + d2 + d3 + duration.count();
    return res;
}

#define N 1000
#define BALANCE 2

int main()
{
    std::vector<int> fa(N);
    std::generate(fa.begin(), fa.end(), std::rand);
    std::vector<int> fb(N);
    std::generate(fb.begin(), fb.end(), std::rand);
    freopen("out.txt", "w", stdout);
    auto multiplier = FFT();
    for(int threads = 1; threads <= 1024; threads++){

        int d;
        // cuda
        auto result_parallel = cumult(fa, fb, BALANCE, threads, d);

      
        cout << threads << " " << d << " ";

        auto start = high_resolution_clock::now(); 
        // sequential
        auto result_sequential = multiplier.mult(fa, fb);

        auto stop = high_resolution_clock::now(); 
        auto duration = duration_cast<microseconds>(stop - start); 

        cout << duration.count() << endl;
       
    }
    
    
    return 0;
}
