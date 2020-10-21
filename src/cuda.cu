#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "cuda.h"
}

__global__ void network_num(int index){
    int sum = index+index;
}