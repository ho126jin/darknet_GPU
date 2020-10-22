#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "cuda.h"
}

__global__ void network_num_kernel(int index){
    int i =0;
    int sum =0;
    for(i=0;i<index*100;i++){
        sum *= index;
    }
}

void network_num(int index){
    /*if(index == 0)
    network_num_kernel<<<1,1>>>(index);
    else if(index == 1)
    network_num_kernel<<<2,2>>>(index);
    else if(index == 2)
    network_num_kernel<<<3,3>>>(index);
    else if(index == 3)
    network_num_kernel<<<4,4>>>(index);
    else if(index == 2)
    network_num_kernel<<<2,2>>>(index);
    else if(index == 2)
    network_num_kernel<<<2,2>>>(index);*/
    network_num_kernel<<<cuda_grid_test(index),1>>>(index);
    check_error(cudaPeekAtLastError());
}