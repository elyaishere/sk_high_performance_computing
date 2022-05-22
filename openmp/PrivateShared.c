#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

/* compile with gcc -o PrivateShared -fopenmp PrivateShared.c */

int main(int argc, char** argv)
{
    int i = 0;
    int tid;
    int size = 20;

    int * a = (int *)calloc(size, sizeof(int));
    int * b = (int *)calloc(size, sizeof(int));
    int * c;
    int * tids = (int *)calloc(size, sizeof(int));

    printf("BEFORE\n");

    for (i = 0; i < size; ++i)
    {
        a[i] = b[i] = i;

        printf("a[%d] = %d, b[%d] = %d\n", i, a[i], i, b[i]);
    }

#pragma omp parallel shared(a, b, tids) private(c, i, tid)
    {
        tid = omp_get_thread_num();

        c = (int *)malloc(sizeof(int));

#pragma omp for
        for (i = 0; i < size; ++i)
        {
            c[0] = tid * a[i];
            a[i] = c[0];
            b[i] += c[0];
            tids[i] = tid;
        }

        free(c);
    }

    printf("AFTER\n");

    for (i = 0; i < size; ++i)
    {
        printf("tid = %d, a[%d] = %d, b[%d] = %d\n", tids[i], i, a[i], i, b[i]);
    }

    free(a);
    free(b);
    free(tids);

    return 0;
}