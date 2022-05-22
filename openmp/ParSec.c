#include <omp.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    const size_t N = 20;

    int i;

    float a[N], b[N], c[N], d[N];

#pragma omp parallel
    {
#pragma omp for
        for (i = 0; i < N; ++i)
        {
            a[i] = i;
            b[i] = N - i;
        }
    }

#pragma omp parallel shared(a,b,c,d) private(i)
    {
#pragma omp sections nowait
        {
#pragma omp section
            for (i = 0; i < N; ++i)
            {
                c[i] = a[i] + b[i];
            }

#pragma omp section
            for (i = 0; i < N; ++i)
            {
                d[i] = a[i] * b[i];
            }
        }
    }

    for (i = 0; i < N; ++i)
    {
        printf("c[%1$d] = %$2f, d[%1$d] = %3$f\n", i, c[i], d[i]); 
    }

    return 0;
 }