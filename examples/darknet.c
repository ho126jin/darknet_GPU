#include "darknet.h"

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>

extern void predict_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top);
extern void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen);
extern void run_yolo(int argc, char **argv);
extern void run_detector(int argc, char **argv);
extern void run_coco(int argc, char **argv);
extern void run_nightmare(int argc, char **argv);
extern void run_classifier(int argc, char **argv);
extern void run_regressor(int argc, char **argv);
extern void run_segmenter(int argc, char **argv);
extern void run_isegmenter(int argc, char **argv);
extern void run_char_rnn(int argc, char **argv);
extern void run_tag(int argc, char **argv);
extern void run_cifar(int argc, char **argv);
extern void run_go(int argc, char **argv);
extern void run_art(int argc, char **argv);
extern void run_super(int argc, char **argv);
extern void run_lsd(int argc, char **argv);
extern void predict_classifier2(test *input);

void average(int argc, char *argv[])
{
    char *cfgfile = argv[2];
    char *outfile = argv[3];
    gpu_index = -1;
    network *net = parse_network_cfg(cfgfile);
    network *sum = parse_network_cfg(cfgfile);

    char *weightfile = argv[4];
    load_weights(sum, weightfile);

    int i, j;
    int n = argc - 5;
    for (i = 0; i < n; ++i)
    {
        weightfile = argv[i + 5];
        load_weights(net, weightfile);
        for (j = 0; j < net->n; ++j)
        {
            layer l = net->layers[j];
            layer out = sum->layers[j];
            if (l.type == CONVOLUTIONAL)
            {
                int num = l.n * l.c * l.size * l.size;
                axpy_cpu(l.n, 1, l.biases, 1, out.biases, 1);
                axpy_cpu(num, 1, l.weights, 1, out.weights, 1);
                if (l.batch_normalize)
                {
                    axpy_cpu(l.n, 1, l.scales, 1, out.scales, 1);
                    axpy_cpu(l.n, 1, l.rolling_mean, 1, out.rolling_mean, 1);
                    axpy_cpu(l.n, 1, l.rolling_variance, 1, out.rolling_variance, 1);
                }
            }
            if (l.type == CONNECTED)
            {
                axpy_cpu(l.outputs, 1, l.biases, 1, out.biases, 1);
                axpy_cpu(l.outputs * l.inputs, 1, l.weights, 1, out.weights, 1);
            }
        }
    }
    n = n + 1;
    for (j = 0; j < net->n; ++j)
    {
        layer l = sum->layers[j];
        if (l.type == CONVOLUTIONAL)
        {
            int num = l.n * l.c * l.size * l.size;
            scal_cpu(l.n, 1. / n, l.biases, 1);
            scal_cpu(num, 1. / n, l.weights, 1);
            if (l.batch_normalize)
            {
                scal_cpu(l.n, 1. / n, l.scales, 1);
                scal_cpu(l.n, 1. / n, l.rolling_mean, 1);
                scal_cpu(l.n, 1. / n, l.rolling_variance, 1);
            }
        }
        if (l.type == CONNECTED)
        {
            scal_cpu(l.outputs, 1. / n, l.biases, 1);
            scal_cpu(l.outputs * l.inputs, 1. / n, l.weights, 1);
        }
    }
    save_weights(sum, outfile);
}

long numops(network *net)
{
    int i;
    long ops = 0;
    for (i = 0; i < net->n; ++i)
    {
        layer l = net->layers[i];
        if (l.type == CONVOLUTIONAL)
        {
            ops += 2l * l.n * l.size * l.size * l.c / l.groups * l.out_h * l.out_w;
        }
        else if (l.type == CONNECTED)
        {
            ops += 2l * l.inputs * l.outputs;
        }
        else if (l.type == RNN)
        {
            ops += 2l * l.input_layer->inputs * l.input_layer->outputs;
            ops += 2l * l.self_layer->inputs * l.self_layer->outputs;
            ops += 2l * l.output_layer->inputs * l.output_layer->outputs;
        }
        else if (l.type == GRU)
        {
            ops += 2l * l.uz->inputs * l.uz->outputs;
            ops += 2l * l.uh->inputs * l.uh->outputs;
            ops += 2l * l.ur->inputs * l.ur->outputs;
            ops += 2l * l.wz->inputs * l.wz->outputs;
            ops += 2l * l.wh->inputs * l.wh->outputs;
            ops += 2l * l.wr->inputs * l.wr->outputs;
        }
        else if (l.type == LSTM)
        {
            ops += 2l * l.uf->inputs * l.uf->outputs;
            ops += 2l * l.ui->inputs * l.ui->outputs;
            ops += 2l * l.ug->inputs * l.ug->outputs;
            ops += 2l * l.uo->inputs * l.uo->outputs;
            ops += 2l * l.wf->inputs * l.wf->outputs;
            ops += 2l * l.wi->inputs * l.wi->outputs;
            ops += 2l * l.wg->inputs * l.wg->outputs;
            ops += 2l * l.wo->inputs * l.wo->outputs;
        }
    }
    return ops;
}

void speed(char *cfgfile, int tics)
{
    if (tics == 0)
        tics = 1000;
    network *net = parse_network_cfg(cfgfile);
    set_batch_network(net, 1);
    int i;
    double time = what_time_is_it_now();
    image im = make_image(net->w, net->h, net->c * net->batch);
    for (i = 0; i < tics; ++i)
    {
        network_predict(net, im.data);
    }
    double t = what_time_is_it_now() - time;
    long ops = numops(net);
    printf("\n%d evals, %f Seconds\n", tics, t);
    printf("Floating Point Operations: %.2f Bn\n", (float)ops / 1000000000.);
    printf("FLOPS: %.2f Bn\n", (float)ops / 1000000000. * tics / t);
    printf("Speed: %f sec/eval\n", t / tics);
    printf("Speed: %f Hz\n", tics / t);
}

void operations(char *cfgfile)
{
    gpu_index = -1;
    network *net = parse_network_cfg(cfgfile);
    long ops = numops(net);
    printf("Floating Point Operations: %ld\n", ops);
    printf("Floating Point Operations: %.2f Bn\n", (float)ops / 1000000000.);
}

void oneoff(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network *net = parse_network_cfg(cfgfile);
    int oldn = net->layers[net->n - 2].n;
    int c = net->layers[net->n - 2].c;
    scal_cpu(oldn * c, .1, net->layers[net->n - 2].weights, 1);
    scal_cpu(oldn, 0, net->layers[net->n - 2].biases, 1);
    net->layers[net->n - 2].n = 11921;
    net->layers[net->n - 2].biases += 5;
    net->layers[net->n - 2].weights += 5 * c;
    if (weightfile)
    {
        load_weights(net, weightfile);
    }
    net->layers[net->n - 2].biases -= 5;
    net->layers[net->n - 2].weights -= 5 * c;
    net->layers[net->n - 2].n = oldn;
    printf("%d\n", oldn);
    layer l = net->layers[net->n - 2];
    copy_cpu(l.n / 3, l.biases, 1, l.biases + l.n / 3, 1);
    copy_cpu(l.n / 3, l.biases, 1, l.biases + 2 * l.n / 3, 1);
    copy_cpu(l.n / 3 * l.c, l.weights, 1, l.weights + l.n / 3 * l.c, 1);
    copy_cpu(l.n / 3 * l.c, l.weights, 1, l.weights + 2 * l.n / 3 * l.c, 1);
    *net->seen = 0;
    save_weights(net, outfile);
}

void oneoff2(char *cfgfile, char *weightfile, char *outfile, int l)
{
    gpu_index = -1;
    network *net = parse_network_cfg(cfgfile);
    if (weightfile)
    {
        load_weights_upto(net, weightfile, 0, net->n);
        load_weights_upto(net, weightfile, l, net->n);
    }
    *net->seen = 0;
    save_weights_upto(net, outfile, net->n);
}

void partial(char *cfgfile, char *weightfile, char *outfile, int max)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 1);
    save_weights_upto(net, outfile, max);
}

void print_weights(char *cfgfile, char *weightfile, int n)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 1);
    layer l = net->layers[n];
    int i, j;
    //printf("[");
    for (i = 0; i < l.n; ++i)
    {
        //printf("[");
        for (j = 0; j < l.size * l.size * l.c; ++j)
        {
            //if(j > 0) printf(",");
            printf("%g ", l.weights[i * l.size * l.size * l.c + j]);
        }
        printf("\n");
        //printf("]%s\n", (i == l.n-1)?"":",");
    }
    //printf("]");
}

void rescale_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    int i;
    for (i = 0; i < net->n; ++i)
    {
        layer l = net->layers[i];
        if (l.type == CONVOLUTIONAL)
        {
            rescale_weights(l, 2, -.5);
            break;
        }
    }
    save_weights(net, outfile);
}

void rgbgr_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    int i;
    for (i = 0; i < net->n; ++i)
    {
        layer l = net->layers[i];
        if (l.type == CONVOLUTIONAL)
        {
            rgbgr_weights(l);
            break;
        }
    }
    save_weights(net, outfile);
}

void reset_normalize_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    int i;
    for (i = 0; i < net->n; ++i)
    {
        layer l = net->layers[i];
        if (l.type == CONVOLUTIONAL && l.batch_normalize)
        {
            denormalize_convolutional_layer(l);
        }
        if (l.type == CONNECTED && l.batch_normalize)
        {
            denormalize_connected_layer(l);
        }
        if (l.type == GRU && l.batch_normalize)
        {
            denormalize_connected_layer(*l.input_z_layer);
            denormalize_connected_layer(*l.input_r_layer);
            denormalize_connected_layer(*l.input_h_layer);
            denormalize_connected_layer(*l.state_z_layer);
            denormalize_connected_layer(*l.state_r_layer);
            denormalize_connected_layer(*l.state_h_layer);
        }
    }
    save_weights(net, outfile);
}

layer normalize_layer(layer l, int n)
{
    int j;
    l.batch_normalize = 1;
    l.scales = calloc(n, sizeof(float));
    for (j = 0; j < n; ++j)
    {
        l.scales[j] = 1;
    }
    l.rolling_mean = calloc(n, sizeof(float));
    l.rolling_variance = calloc(n, sizeof(float));
    return l;
}

void normalize_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    int i;
    for (i = 0; i < net->n; ++i)
    {
        layer l = net->layers[i];
        if (l.type == CONVOLUTIONAL && !l.batch_normalize)
        {
            net->layers[i] = normalize_layer(l, l.n);
        }
        if (l.type == CONNECTED && !l.batch_normalize)
        {
            net->layers[i] = normalize_layer(l, l.outputs);
        }
        if (l.type == GRU && l.batch_normalize)
        {
            *l.input_z_layer = normalize_layer(*l.input_z_layer, l.input_z_layer->outputs);
            *l.input_r_layer = normalize_layer(*l.input_r_layer, l.input_r_layer->outputs);
            *l.input_h_layer = normalize_layer(*l.input_h_layer, l.input_h_layer->outputs);
            *l.state_z_layer = normalize_layer(*l.state_z_layer, l.state_z_layer->outputs);
            *l.state_r_layer = normalize_layer(*l.state_r_layer, l.state_r_layer->outputs);
            *l.state_h_layer = normalize_layer(*l.state_h_layer, l.state_h_layer->outputs);
            net->layers[i].batch_normalize = 1;
        }
    }
    save_weights(net, outfile);
}

void statistics_net(char *cfgfile, char *weightfile)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    int i;
    for (i = 0; i < net->n; ++i)
    {
        layer l = net->layers[i];
        if (l.type == CONNECTED && l.batch_normalize)
        {
            printf("Connected Layer %d\n", i);
            statistics_connected_layer(l);
        }
        if (l.type == GRU && l.batch_normalize)
        {
            printf("GRU Layer %d\n", i);
            printf("Input Z\n");
            statistics_connected_layer(*l.input_z_layer);
            printf("Input R\n");
            statistics_connected_layer(*l.input_r_layer);
            printf("Input H\n");
            statistics_connected_layer(*l.input_h_layer);
            printf("State Z\n");
            statistics_connected_layer(*l.state_z_layer);
            printf("State R\n");
            statistics_connected_layer(*l.state_r_layer);
            printf("State H\n");
            statistics_connected_layer(*l.state_h_layer);
        }
        printf("\n");
    }
}

void denormalize_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network *net = load_network(cfgfile, weightfile, 0);
    int i;
    for (i = 0; i < net->n; ++i)
    {
        layer l = net->layers[i];
        if ((l.type == DECONVOLUTIONAL || l.type == CONVOLUTIONAL) && l.batch_normalize)
        {
            denormalize_convolutional_layer(l);
            net->layers[i].batch_normalize = 0;
        }
        if (l.type == CONNECTED && l.batch_normalize)
        {
            denormalize_connected_layer(l);
            net->layers[i].batch_normalize = 0;
        }
        if (l.type == GRU && l.batch_normalize)
        {
            denormalize_connected_layer(*l.input_z_layer);
            denormalize_connected_layer(*l.input_r_layer);
            denormalize_connected_layer(*l.input_h_layer);
            denormalize_connected_layer(*l.state_z_layer);
            denormalize_connected_layer(*l.state_r_layer);
            denormalize_connected_layer(*l.state_h_layer);
            l.input_z_layer->batch_normalize = 0;
            l.input_r_layer->batch_normalize = 0;
            l.input_h_layer->batch_normalize = 0;
            l.state_z_layer->batch_normalize = 0;
            l.state_r_layer->batch_normalize = 0;
            l.state_h_layer->batch_normalize = 0;
            net->layers[i].batch_normalize = 0;
        }
    }
    save_weights(net, outfile);
}

void mkimg(char *cfgfile, char *weightfile, int h, int w, int num, char *prefix)
{
    network *net = load_network(cfgfile, weightfile, 0);
    image *ims = get_weights(net->layers[0]);
    int n = net->layers[0].n;
    int z;
    for (z = 0; z < num; ++z)
    {
        image im = make_image(h, w, 3);
        fill_image(im, .5);
        int i;
        for (i = 0; i < 100; ++i)
        {
            image r = copy_image(ims[rand() % n]);
            rotate_image_cw(r, rand() % 4);
            random_distort_image(r, 1, 1.5, 1.5);
            int dx = rand() % (w - r.w);
            int dy = rand() % (h - r.h);
            ghost_image(r, im, dx, dy);
            free_image(r);
        }
        char buff[256];
        sprintf(buff, "%s/gen_%d", prefix, z);
        save_image(im, buff);
        free_image(im);
    }
}

void visualize(char *cfgfile, char *weightfile)
{
    network *net = load_network(cfgfile, weightfile, 0);
    visualize_network(net);
}

void choiceNetwork()
{
}

//twin_thpool *twin_thp;
threadpool thpool;
//각 네트워크의 조건변수, mutex변수, wait를 위한 변수 선언 헤더에 extern변수로 지정
pthread_cond_t *cond_t;
pthread_mutex_t *mutex_t;
int *cond_i;

//
#define n_net 8 //hojin 8->2

//hojin each networknum
#define n_des 16
#define n_res 0
#define n_alex 0
#define n_vgg 0

#define THRESHOLD 4

double gpu_total_time = 0;
#define cpu_thread 0
#define gpu_thread 8
//kmsjames 2020 0819 for pwr mon
static int pwr_ind=0;
static int pwr_ind_finish =0;
// void *pwr_mon_do(void)
// {
// 	char buffer[10];
// 	int fd[6];
// 	int val[6];
// 	int total_pwr=0;

//         FILE *fp = fopen("total_pwr_cpu_gpu.txt","a");

// 	while(1){
// 		if(pwr_ind == 0) continue;
	
// 		total_pwr = 0;	
// 		fd[0] = open("/sys/bus/i2c/drivers/ina3221x/1-0040/iio_device/in_power0_input", O_RDONLY);
// 		read(fd[0], buffer, 5);
// 	 	val[0] = atoi(buffer);
		
// 		fd[1] = open("/sys/bus/i2c/drivers/ina3221x/1-0040/iio_device/in_power1_input", O_RDONLY);
// 		read(fd[1], buffer, 5);
// 	 	val[1] = atoi(buffer);
		
// 		fd[2] = open("/sys/bus/i2c/drivers/ina3221x/1-0040/iio_device/in_power2_input", O_RDONLY);
// 		read(fd[2], buffer, 5);
// 	 	val[2] = atoi(buffer);
		
// 		fd[3] = open("/sys/bus/i2c/drivers/ina3221x/1-0041/iio_device/in_power0_input", O_RDONLY);
// 		read(fd[3], buffer, 5);
// 	 	val[3] = atoi(buffer);
		
// 		fd[4] = open("/sys/bus/i2c/drivers/ina3221x/1-0041/iio_device/in_power1_input", O_RDONLY);
// 		read(fd[4], buffer, 5);
// 	 	val[4] = atoi(buffer);
		
// 		fd[5] = open("/sys/bus/i2c/drivers/ina3221x/1-0041/iio_device/in_power2_input", O_RDONLY);
// 		read(fd[5], buffer, 5);
// 	 	val[5] = atoi(buffer);
		
// 		for(int nu=0; nu<6;nu++){
// 			total_pwr += val[nu];	
// 			close(fd[nu]);
// 		}	

// 		if (fp)
//     		{
// 		        for(int nu =0; nu<6; nu++) fprintf(fp, " %d,",val[nu]);
// 			fprintf(fp, "\n");
// 			fprintf(fp, "total_pwr = %d\n", total_pwr);
//     		}
//     		else
//     		{
//         		fprintf(stderr, "file open error");
//         		exit(1);
//     		}
// 		if(pwr_ind_finish == 1) break;

// 		usleep(10);
// 	}
// 	fclose(fp);	
// }
//FILE * timing;

int main()
{
#ifndef GPU
    gpu_index = -1;
#else
    gpu_index = 0;
    if (gpu_index >= 0)
    {
        cuda_set_device(gpu_index);
    }
    cudaSetDeviceFlags(cudaDeviceMapHost);
    //timing = fopen("thread_cpu.txt","a+");
#endif
    int n_all = n_des+n_res+n_vgg+n_alex;
#ifdef THREAD
    //twin_thp = twin_thpool_init(cpu_thread,gpu_thread);
    thpool = thpool_init(gpu_thread,n_all);
#endif

    //char** vgg = {"darknet", "classfier", "predict", "cfg/imagenet1k.data", "cfg/","", "data/eagle.jpg"};
    //char** densenet = {"darknet", "classfier", "predict", "cfg/imagenet1k.data", "cfg/", "", "data/eagle.jpg"};
    //char** resnet = {"darknet", "classfier", "predict", "cfg/imagenet1k.data", "cfg/", "", "data/eagle.jpg"};
    char *alexName = "Alex";
    char *vggName = "VGG";
    char *denseName = "Dense";
    char *resName = "Res";

    network *denseNetwork[n_des];
    network *resNetwork[n_res];
    network *vggNetwork[n_vgg];
    network *alexNetwork[n_alex];

    int i = 0;
#ifdef THREAD
    //변수 동적할당
    cond_t = (pthread_cond_t *)malloc(sizeof(pthread_cond_t) * n_all);
    mutex_t = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t) * n_all);
    cond_i = (int *)malloc(sizeof(int) * n_all);

    for (i = 0; i < n_all; i++)
    {
        pthread_cond_init(&cond_t[i], NULL);
        pthread_mutex_init(&mutex_t[i], NULL);
        cond_i[i] = 0;
    }
#endif
#if 0
    unsigned k = 0;
    for (k = 0; k < n_net; k++)
    {
        denseNetwork[k] = (network *)load_network("cfg/densenet201.cfg", "densenet201.weights", 0);
        denseNetwork[k]->index_n = k;
        resNetwork[k] = (network *)load_network("cfg/resnet152.cfg", "resnet152.weights", 0);
        resNetwork[k]->index_n = k + n_net;
    }
#endif
    int k;
     for (k = 0; k < n_des; k++)
    {
        denseNetwork[k] = (network *)load_network("cfg/densenet201.cfg", "densenet201.weights", 0);
        denseNetwork[k]->index_n = k;
        denseNetwork[k]->priority = k;
    }

    for (k = 0; k < n_res; k++)
    {
        resNetwork[k] = (network *)load_network("cfg/resnet152.cfg", "resnet152.weights", 0);
        resNetwork[k]->index_n = k + n_des;
        resNetwork[k]->priority = k + n_des;
    }
    for (k = 0; k < n_vgg; k++)
    {
        vggNetwork[k] = (network *)load_network("cfg/vgg-16.cfg", "vgg-16.weights", 0);
        vggNetwork[k]->index_n = k + n_des + n_res;
        vggNetwork[k]->priority = k + n_des + n_res;
    }
    for (k = 0; k < n_alex; k++)
    {
        alexNetwork[k] = (network *)load_network("cfg/alexnet.cfg", "alexnet.weights", 0);
        alexNetwork[k]->index_n = k + n_des + n_res + n_vgg;
        alexNetwork[k]->priority = k + n_des + n_res + n_vgg;
    }
//while(1){
//    char c='n';
//    printf("inference start y or n : ");
//    scanf("%c", &c);
//    getchar();
//    if(c == 'n'){
//	    break;
//    }
// **********************stream 자리**************

#ifdef STREAM
    set_stream(n_all,THRESHOLD);
#endif
 
#ifdef CUDNN
    cudnn_handle_set(n_all);
#endif

    list *options = read_data_cfg("cfg/imagenet1k.data");
    char *name_list = option_find_str(options, "names", 0);
    if (!name_list)
        name_list = option_find_str(options, "labels", "data/labels.list");
    int top = option_find_int(options, "top", 1);

    char **names = get_labels(name_list);

    char *buff = "data/eagle.jpg";
    //char *input = buff;
    test *net_input_des[n_des];
    test *net_input_res[n_res];
    test *net_input_vgg[n_vgg];
    test *net_input_alex[n_alex];

    image im = load_image_color(buff, 0, 0);

    double time = what_time_is_it_now();

    pthread_t networkArray_des[n_des];
    pthread_t networkArray_res[n_res];
    pthread_t networkArray_vgg[n_vgg];
    pthread_t networkArray_alex[n_alex];

//kmsjames 2020 0819 for power monitor thread
   /* pthread_t *pwr_mon_thread;
    if (pthread_create(&pwr_mon_thread, NULL, (void *)pwr_mon_do, NULL) < 0)
    {
            perror("pwr_mon thread error");
            exit(0);
    }

*/

    // for(int i=0; i<vggCount; ++i){
    //     test * input = (test*)malloc(sizeof(test));
    //     input->net = *vggNetwork;
    //     input->im = im;
    //     input->names = names;
    //     input->netName = vggName;

    //     pthread_create(&networkArray[count], NULL,predict_classifier2, input);
    //     ++count;
    // }
    //

//kmsjames 2020 0819 for power mon
    pwr_ind = 1;
cudaProfilerStart();

    for (i = 0; i < n_des; i++)
    {
        net_input_des[i] = (test *)malloc(sizeof(test));
        net_input_des[i]->net = denseNetwork[i];
        net_input_des[i]->input_path = buff;
        net_input_des[i]->names = names;
        net_input_des[i]->netName = denseName;
        printf(" It's turn for des i = %d\n", i);
        if (pthread_create(&networkArray_des[i], NULL, (void *)predict_classifier2, net_input_des[i]) < 0)
        {
            perror("thread error");
            exit(0);
        }
    }

    for (i = 0; i < n_res; i++)
    {
        net_input_res[i] = (test *)malloc(sizeof(test));
        net_input_res[i]->net = resNetwork[i];
        net_input_res[i]->input_path = buff;
        net_input_res[i]->names = names;
        net_input_res[i]->netName = resName;

        printf(" It's turn for res i = %d\n", i);
        if (pthread_create(&networkArray_res[i], NULL, (void *)predict_classifier2, net_input_res[i]) < 0)
        {
            perror("thread error");
            exit(0);
        }
    }

    for (i = 0; i < n_vgg; i++)
    {
            net_input_vgg[i] = (test *)malloc(sizeof(test));
            net_input_vgg[i]->net = vggNetwork[i];
            net_input_vgg[i]->input_path = buff;
            net_input_vgg[i]->names = names;
            net_input_vgg[i]->netName = vggName;

        printf(" It's turn for vgg i = %d\n", i);
        if (pthread_create(&networkArray_vgg[i], NULL, (void *)predict_classifier2, net_input_vgg[i]) < 0)
        {
            perror("thread error");
            exit(0);
        }
    }

    for (i = 0; i < n_alex; i++)
    {
            net_input_alex[i] = (test *)malloc(sizeof(test));
            net_input_alex[i]->net = alexNetwork[i];
            net_input_alex[i]->input_path = buff;
            net_input_alex[i]->names = names;
            net_input_alex[i]->netName = alexName;

        printf(" It's turn for alex i = %d\n", i);
        if (pthread_create(&networkArray_alex[i], NULL, (void *)predict_classifier2, net_input_alex[i]) < 0)
        {
            perror("thread error");
            exit(0);
        }
    }
    
    //fprintf(stderr, "\n execution Time : %lf\n", what_time_is_it_now() - time);


    for (i = 0; i < n_des; i++)
    {
        pthread_join(networkArray_des[i], NULL);
    }
    for (i = 0; i < n_res; i++)
    {
        pthread_join(networkArray_res[i], NULL);
    }
    for (i = 0; i < n_vgg; i++)
    {
        pthread_join(networkArray_vgg[i], NULL);
    }
    for (i = 0; i < n_alex; i++)
    {
        pthread_join(networkArray_alex[i], NULL);
    }

#if 0
    for (i = 0; i < n_net; i++)
    {
        pthread_join(networkArray_des[i], NULL);
        pthread_join(networkArray_res[i], NULL);
    }
#endif
#if 0
    //kmsjames 2020 0215
    for(i=0; i<THREAD_NUM_POOL;i++)
	    pthread_join(thpool->threads[i]->pthread, NULL);
#endif
   //FILE *fp =fopen("exetime.txt", "a+");
    fprintf(stderr, "\n execution Time : %lf\n", what_time_is_it_now() - time);
	cudaProfilerStop();
    /*
    if (fp)
    {
        fprintf(fp, "%lf\n", what_time_is_it_now() - time);
    }
    else
    {
        fprintf(stderr, "file open error");
        exit(1);
    }
    fclose(fp);
    */
//kmsjames 2020 0819 for power mon
    pwr_ind_finish = 1;    

//    fclose(fp);
    free(cond_t);
    free(mutex_t);
    free(cond_i);
    return 0;
    /*
    if (0 == strcmp(argv[1], "average")){
        average(argc, argv);
    } else if (0 == strcmp(argv[1], "yolo")){
        run_yolo(argc, argv);
    } else if (0 == strcmp(argv[1], "super")){
        run_super(argc, argv);
    } else if (0 == strcmp(argv[1], "lsd")){
        run_lsd(argc, argv);
    } else if (0 == strcmp(argv[1], "detector")){
        run_detector(argc, argv);
    } else if (0 == strcmp(argv[1], "detect")){
        float thresh = find_float_arg(argc, argv, "-thresh", .5);
        char *filename = (argc > 4) ? argv[4]: 0;
        char *outfile = find_char_arg(argc, argv, "-out", 0);
        int fullscreen = find_arg(argc, argv, "-fullscreen");
        test_detector("cfg/coco.data", argv[2], argv[3], filename, thresh, .5, outfile, fullscreen);
    } else if (0 == strcmp(argv[1], "cifar")){
        run_cifar(argc, argv);
    } else if (0 == strcmp(argv[1], "go")){
        run_go(argc, argv);
    } else if (0 == strcmp(argv[1], "rnn")){
        run_char_rnn(argc, argv);
    } else if (0 == strcmp(argv[1], "coco")){
        run_coco(argc, argv);
    } else if (0 == strcmp(argv[1], "classify")){
        predict_classifier("cfg/imagenet1k.data", argv[2], argv[3], argv[4], 5);
    } else if (0 == strcmp(argv[1], "classifier")){
        run_classifier(argc, argv);
    } else if (0 == strcmp(argv[1], "regressor")){
        run_regressor(argc, argv);
    } else if (0 == strcmp(argv[1], "isegmenter")){
        run_isegmenter(argc, argv);
    } else if (0 == strcmp(argv[1], "segmenter")){
        run_segmenter(argc, argv);
    } else if (0 == strcmp(argv[1], "art")){
        run_art(argc, argv);
    } else if (0 == strcmp(argv[1], "tag")){
        run_tag(argc, argv);
    } else if (0 == strcmp(argv[1], "3d")){
        composite_3d(argv[2], argv[3], argv[4], (argc > 5) ? atof(argv[5]) : 0);
    } else if (0 == strcmp(argv[1], "test")){
        test_resize(argv[2]);
    } else if (0 == strcmp(argv[1], "nightmare")){
        run_nightmare(argc, argv);
    } else if (0 == strcmp(argv[1], "rgbgr")){
        rgbgr_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "reset")){
        reset_normalize_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "denormalize")){
        denormalize_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "statistics")){
        statistics_net(argv[2], argv[3]);
    } else if (0 == strcmp(argv[1], "normalize")){
        normalize_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "rescale")){
        rescale_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "ops")){
        operations(argv[2]);
    } else if (0 == strcmp(argv[1], "speed")){
        speed(argv[2], (argc > 3 && argv[3]) ? atoi(argv[3]) : 0);
    } else if (0 == strcmp(argv[1], "oneoff")){
        oneoff(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "oneoff2")){
        oneoff2(argv[2], argv[3], argv[4], atoi(argv[5]));
    } else if (0 == strcmp(argv[1], "print")){
        print_weights(argv[2], argv[3], atoi(argv[4]));
    } else if (0 == strcmp(argv[1], "partial")){
        partial(argv[2], argv[3], argv[4], atoi(argv[5]));
    } else if (0 == strcmp(argv[1], "average")){
        average(argc, argv);
    } else if (0 == strcmp(argv[1], "visualize")){
        visualize(argv[2], (argc > 3) ? argv[3] : 0);
    } else if (0 == strcmp(argv[1], "mkimg")){
        mkimg(argv[2], argv[3], atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), argv[7]);
    } else if (0 == strcmp(argv[1], "imtest")){
        test_resize(argv[2]);
    } else {
        fprintf(stderr, "Not an option: %s\n", argv[1]);
    }*/
}
