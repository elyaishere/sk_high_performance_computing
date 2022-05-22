#include <omp.h>
#include <iostream>
#include <stdio.h>

int main(int argc, char ** argv)
{
    int tid;

    // C++ cout always bad
#pragma omp parallel private(tid)
    {
        tid = omp_get_thread_num();
        std::cout << "cout " << tid << " hello world" << std::endl;
    }

    // printf sometimes bad, depends on implementation
#pragma omp parallel private(tid)
    {
        tid = omp_get_thread_num();
        printf("printf %d hello world\n", tid);
    }

    // correct way
#pragma omp parallel private(tid)
    {
        tid = omp_get_thread_num();
        
        // force printfs to execute in sequential way    
#pragma omp critical
        {
            printf("critical %d hello world\n", tid);
            /// try to use cout instead printf here
            // std::cout << "cout " << tid << " hello world" << std::endl;
        }
    }

    return 0;
}