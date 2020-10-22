#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "cuda.h"
}

__global__ void network_num_kernel(int index){
    int i =0,j=0;
    int sum =0;
    for(i=0;i<index*1000;i++){
        sum += 1;
        for(j=0;j<1000;j++)
            sum *= index;
    }
}

void network_num(int index,cudaStream_t *stream){
    network_num_kernel<<<cuda_grid_test(index),1,0,*stream>>>(index);
    check_error(cudaPeekAtLastError());
}