#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "cuda.h"
}

__global__ void network_num_kernel(int index){
    int sum = 1;
    int sum += 1;
}

void network_num(int index){
    network_num_kernel<<<index,index>>>(index);
    check_error(cudaPeekAtLastError());
}