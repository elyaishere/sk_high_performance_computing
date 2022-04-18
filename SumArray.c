#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char ** argv)
{
    const size_t N = 10000;

    int a[N];

#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        a[i] = i;
    }

    int sum = 0;

#pragma omp parallel for reduction(+: sum)
    for (int i = 0; i < N; ++i)
    {
        sum += a[i];
    }

    printf("sum = %d\n", sum);

    return 0;
}