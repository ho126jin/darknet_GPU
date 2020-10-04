#include "time_checker.h"
//lcs 0815

inline double convTime(int c, int size, int out_w, int out_h, int out_c)
{
    return c * size * size * out_w * out_h * out_c * 3.07709e-10 + 0.0044042;
}
inline double convTime_gpu(int c, int size, int out_w, int out_h, int out_c)
{
    return c * size * size * out_w * out_h * out_c * 1.79984e-12 + 0.0001643;
}
inline double avgTime(int w, int h, int c)
{
    return w * h * c * 2.54309896e-09;
}
inline double avgTime_gpu(int w, int h, int c)
{
    return w * h * c * 4.71640625e-10;
}
inline double connectTime(int inputs, int outputs)
{
    return inputs * 1.41e-06 + outputs * 3.73e-7 + 0.0038565;
}
inline double connectTime_gpu(int inputs, int outputs)
{
    return inputs * 1.55e-7 + outputs * 1.08e-7 + 3.30e-04;
}
inline double maxTime(int size, int out_w, int out_h, int out_c)
{
    return size * 6.49e-04 + out_w * 6.72e-5 + out_h * 6.72e-5 + out_c * 5.42e-6 - 3.73e-3;
}
inline double maxTime_gpu(int size, int out_w, int out_h, int out_c)
{
    return size * 1.22e-5 + out_w * 1.14e-6 + out_h * 1.14e-6 + out_c * 1.09e-7 - 2.24e-5;
}
inline double routeTime(int input1, int input2)
{
    return (double)input1 * -1.11e-9 + (double)input2 * 2.24e-09 + 0.0007599;
}
inline double routeTime_gpu(int input1, int input2)
{
    return (double)input1 * -4.96893e-10 + (double)input2 * 7.1213e-11 + 7.65e-5;
}

inline double shortTime(int w1, int w2, int h1, int h2, int c1, int c2)
{
    int minw = (w1 > w2) ? w2 : w1;
    int minh = (h1 > h2) ? h2 : h1;
    int minc = (c1 > c2) ? c2 : c1;

    return minw * minh * minc * 9.53e-9 + 0.0005095;
}
inline double shortTime_gpu(int w1, int w2, int h1, int h2, int c1, int c2)
{
    int minw = (w1 > w2) ? w2 : w1;
    int minh = (h1 > h2) ? h2 : h1;
    int minc = (c1 > c2) ? c2 : c1;

    return minw * minh * minc * 2.36191e-10;
}
inline double softTime(int inputs)
{
    return inputs * 6.424e-08;
}
inline double softTime_gpu(int inputs)
{
    return inputs * 7.07866667e-08;
}
