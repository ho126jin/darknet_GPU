//gpuindex cuda.c
int gpu_index = 0;

#ifdef GPU

#include "cuda.h"
#include "utils.h"
#include "blas.h"
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_profiler_api.h"

static int check = 0;

void cuda_set_device(int n)
{
   //if(check == 0){
    gpu_index = n;
    cudaError_t status = cudaSetDevice(n);
    check_error(status);
    check = 1;
  // }
}

int cuda_get_device()
{
    int n = 0;
    cudaError_t status = cudaGetDevice(&n);
    check_error(status);
    return n;
}

void check_error(cudaError_t status)
{
    //cudaDeviceSynchronize();
    cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error: %s", s);
        error(buffer);
    }
    if (status2 != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error Prev: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
        error(buffer);
    }
}

//cs 0904 stream
void check_error_line(cudaError_t status, int line)
{
    //cudaDeviceSynchronize();
    cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error: %s, LINE : %d\n", s, line);
        assert(0);
        snprintf(buffer, 256, "CUDA Error: %s", s);
        error(buffer);
    }
    if (status2 != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error Prev: %s, LINE : %d\n", s, line);
        assert(0);
        snprintf(buffer, 256, "CUDA Error Prev: %s, LINE : %d\n", s, line);
        error(buffer);
    }
}

#ifdef STREAM
    static cudaStream_t stream[n_a];
    static int init_stream[n_a] ={ 0, };
    void set_stream(int num,int threshold){
        int i;
        int priority_high, priority_low;
        cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);

        for(i=0;i<num;i++)
        // create streams with highest and lowest available priorities
        if(i>=threshold){
            cudaStreamCreateWithPriority(&(stream[i]), cudaStreamNonBlocking, priority_high);
            network_num(i);
	} else{
            cudaStreamCreateWithPriority(&(stream[i]), cudaStreamNonBlocking, priority_low);
            network_num(i);
	}
    }
    
#endif

#ifdef CUDNN
    static cudnnHandle_t handle[n_a];
    static int init_[n_a] ={0,};
    void cudnn_handle_set(int num)
    {
        int i;
        for (i = 0; i < num; i++)
        {
            cudnnCreate(&handle[i]);
            #ifdef STREAM    
                cudaError_t status = cudnnSetStream(handle[i], stream[i]);
                check_error(status);
            #endif
            //cudaStreamCreate(&(stream[i]));
            init_[i] = 1;
        }
    }
    
        cudnnHandle_t cudnn_handle(int id)
        {
            #ifdef STREAM
                int i = id;
                if(!init_[i]){
                        cudnnCreate(&handle[i]);
                        cudaStreamCreateWithFlags(&(stream[i]), cudaStreamNonBlocking);
                        //cudaStreamCreate(&(stream[i]));
                        cudaError_t status = cudnnSetStream(handle[i], stream[i]);
                        init_[i] = 1;
                    }
            #else
                int i = cuda_get_device();
                if (!init_[i])
                {
                    cudnnCreate(&handle[i]);
                    init_[i] = 1;
                }
            #endif
            return handle[i];
        }
#endif


void cudnn_handle_set_stream(int num)
{
    int i;
    for (i = 0; i < num; i++)
    {
        cudaStreamCreateWithFlags(&(stream[i]), cudaStreamNonBlocking);
        //cudaStreamCreate(&(stream[i]));
        init_stream[i] = 1;
    }
}

void cuda_synchronize(int id, int line)
{
    cudaError_t status = cudaStreamSynchronize(stream[id]);
    check_error_line(status, line);
}
cudaStream_t usedstream(int id)
{
    return stream[id];
}

void cuda_pull_array_stream(float *x_gpu, float *x, size_t n, int id)
{
    size_t size = sizeof(float) * n;
    cudaError_t status = cudaMemcpyAsync(x, x_gpu, size, cudaMemcpyDeviceToHost, stream[id]);
    check_error(status);
}

void cuda_push_array_stream(float *x_gpu, float *x, size_t n, int id)
{
    size_t size = sizeof(float) * n;
    cudaError_t status = cudaMemcpyAsync(x_gpu, x, size, cudaMemcpyHostToDevice, stream[id]);
    check_error(status);
}


dim3 cuda_gridsize(size_t n)
{
    size_t k = (n - 1) / BLOCK + 1;
    size_t x = k;
    size_t y = 1;
    if (x > 65535)
    {
        x = ceil(sqrt(k));
        y = (n - 1) / (x * BLOCK) + 1;
    }
    dim3 d ={ x, y, 1 };
    //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
    return d;
}

dim3 cuda_grid_test(size_t n)
{
    size_t x = n+1;
    size_t y = n+1;
    dim3 d = {x,y,1};
    return d;
}

static int init_blas[32] ={ 0 };
static cublasHandle_t handle_blas[32];

cublasHandle_t blas_handle_a(int idx)
{
    int i = idx;
    if (!init_blas[i])
    {
        cublasCreate(&handle_blas[i]);
        #ifdef STREAM
            cublasSetStream(handle_blas[i], stream[i]);
        #endif
        init_blas[i] = 1;
    }

    return handle_blas[i];
}





cublasHandle_t blas_handle()
{
    static int init[16] ={ 0 };
    static cublasHandle_t handle[16];
    int i = cuda_get_device();
    if (!init[i])
    {
        cublasCreate(&handle[i]);
        init[i] = 1;
    }
    return handle[i];
}

float *cuda_make_array(float *x, size_t n)
{
    float *x_gpu;
    size_t size = sizeof(float) * n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    check_error(status);
//     #ifdef STREAM
//     //2020 0311 doyoung
//     cudaMemset(x_gpu, .0, size);
// #endif
    if (x)
    {
        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        check_error(status);
    }
    else
    {
        fill_gpu(n, 0, x_gpu, 1);
    }
    if (!x_gpu)
        error("Cuda malloc failed\n");
    return x_gpu;
}

float *cuda_make_array_2(float *x, size_t n)
{
    float *x_gpu;
    size_t size = sizeof(float) * n;
    cudaError_t status; // = cudaMalloc((void **)&x_gpu, size);

    status = cudaHostGetDevicePointer((void **)&x_gpu, (void *)x, 0);

    check_error(status);
    if (x)
    {
        //status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        check_error(status);
    }
    else
    {

        fill_gpu(n, 0, x_gpu, 1);
    }
    if (!x_gpu)
        error("Cuda malloc failed\n");
    return x_gpu;
    // float *x_gpu;
    // size_t size = sizeof(float) * n;
    // cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    // check_error(status);
    // if (x)
    // {
    //     status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
    //     check_error(status);
    // }
    // else
    // {
    //     fill_gpu(n, 0, x_gpu, 1);
    // }
    // if (!x_gpu)
    //     error("Cuda malloc failed\n");
    // return x_gpu;
}

void cuda_random(float *x_gpu, size_t n)
{
    static curandGenerator_t gen[16];
    static int init[16] ={ 0 };
    int i = cuda_get_device();
    if (!init[i])
    {
        curandCreateGenerator(&gen[i], CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen[i], time(0));
        init[i] = 1;
    }
    curandGenerateUniform(gen[i], x_gpu, n);
    check_error(cudaPeekAtLastError());
}

float cuda_compare(float *x_gpu, float *x, size_t n, char *s)
{
    float *tmp = calloc(n, sizeof(float));
    cuda_pull_array(x_gpu, tmp, n);
    //int i;
    //for(i = 0; i < n; ++i) printf("%f %f\n", tmp[i], x[i]);
    axpy_cpu(n, -1, x, 1, tmp, 1);
    float err = dot_cpu(n, tmp, 1, tmp, 1);
    printf("Error %s: %f\n", s, sqrt(err / n));
    free(tmp);
    return err;
}

int *cuda_make_int_array(int *x, size_t n)
{
    int *x_gpu;
    size_t size = sizeof(int) * n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    check_error(status);
    if (x)
    {
        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        check_error(status);
    }
    if (!x_gpu)
        error("Cuda malloc failed\n");
    return x_gpu;
}

void cuda_free(float *x_gpu)
{
    cudaError_t status = cudaFree(x_gpu);
    check_error(status);
}

void cuda_push_array(float *x_gpu, float *x, size_t n)
{
    size_t size = sizeof(float) * n;
    cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
    check_error(status);
}

void cuda_pull_array(float *x_gpu, float *x, size_t n)
{
    size_t size = sizeof(float) * n;
    cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);
    check_error(status);
}

float cuda_mag_array(float *x_gpu, size_t n)
{
    float *temp = calloc(n, sizeof(float));
    cuda_pull_array(x_gpu, temp, n);
    float m = mag_array(temp, n);
    free(temp);
    return m;
}
#else
void cuda_set_device(int n)
{
}

#endif
