#ifndef THPOOL_EX_H
#define THPOOL_EX_H

typedef struct twin_thpool
{
    threadpool thpool_cpu;
    threadpool thpool_gpu;
} twin_thpool;

twin_thpool *twin_thpool_init(int, int);

int add_job(twin_thpool *, void (*function)(void *), netlayer *arg_p, int flag, int *routeOrShort);

#endif