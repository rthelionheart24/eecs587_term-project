#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include "cuda_runtime.h"

#include "kernel.cu"

using namespace std;

int main(int argc, char* argv[]) {

    int *h_A;
    int *d_A;

    h_A = (int*)malloc(sizeof(int));
    *h_A = 1;

    cudaMalloc((void**)&d_A, sizeof(int));

    cudaMemcpy(d_A, h_A, sizeof(int), cudaMemcpyHostToDevice);

    executeTask<<<1, 1>>>(d_A);

    cudaMemcpy(h_A, d_A, sizeof(int), cudaMemcpyDeviceToHost);


    cudafree(d_A);
    free(h_A);

    return 0;

}