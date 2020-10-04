#ifndef TIME_CHECKER_H
#define TIME_CHECKER_H

extern double convTime(int c, int size, int out_w, int out_h, int out_c);
extern double convTime_gpu(int c, int size, int out_w, int out_h, int out_c);
extern double avgTime(int w, int h, int c);
extern double avgTime_gpu(int w, int h, int c);
extern double connectTime(int inputs, int outputs);
extern double connectTime_gpu(int inputs, int outputs);
extern double maxTime(int size, int out_w, int out_h, int out_c);
extern double maxTime_gpu(int size, int out_w, int out_h, int out_c);
extern double routeTime(int input1, int input2);
extern double routeTime_gpu(int intput1, int input2);
extern double shortTime(int w1, int w2, int h1, int h2, int c1, int c2);
extern double shortTime_gpu(int w1, int w2, int h1, int h2, int c1, int c2);
extern double softTime(int inputs);
extern double softTime_gpu(int inputs);

#endif
