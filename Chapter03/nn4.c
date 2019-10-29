#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#define HIDDENNO 6      // 隐含层Cell数
#define OUTPUTNO 4      // 输出层Cell数
#define HALF_RAND_MAX (RAND_MAX / 2)
#define ALPHA 10        // 学习系数
#define SEED 65535      // 随机数SEED
#define MAXINPUTNO 50   // 学习数据的最大个数
#define MAX_TRAIN_TIMES 500
#define BIGNUM 100      // 误差初始值
#define LIMIT 0.001     // 误差上限值
#define IMAGESIZE 8     // 输入图像边长
#define F_SIZE 3        // 卷积过滤器的大小
#define F_NO 2          // 卷积过滤器的数量
#define POOLOUTSIZE 3   // 池化层大小
#define POOLSIZE 2      // 池化区域大小 注意  POOLSIZE * POOLOUTSIZE = IMAGESIZE + 1 - F_SIZE
#define INPUTNO (POOLOUTSIZE * POOLOUTSIZE * F_NO)        // 输入层Cell数
#define _CRT_SECURE_NO_WARNINGS

void conv(double filter[][F_SIZE], double e[][IMAGESIZE], double convout[][IMAGESIZE]);     // 卷积函数
double calcconv(double filter[][F_SIZE], double e[][IMAGESIZE], int i, int j);
void pool(double convout[][IMAGESIZE], double poolout[][POOLOUTSIZE]);
double calcpooling(double convout[][IMAGESIZE], int i, int j);


void conv(double filter[][F_SIZE], double e[][IMAGESIZE], double convout[][IMAGESIZE])
{   
    int startpoint = F_SIZE / 2;
    for( int i = 0; i < IMAGESIZE - startpoint; ++i)
        for( int j = 0; j < IMAGESIZE - startpoint; ++j)
            convout[i][j] = calcconv(filter, e, i, j);
}


double calcconv(double filter[][F_SIZE], double e[][IMAGESIZE], int i, int j)
{   // 卷积
    int i_start = i - F_SIZE / 2;
    int j_start = j - F_SIZE / 2;
    double sum = 0.0;
    for( int m = 0; m < F_SIZE; ++m)
        for( int n = 0; n < F_SIZE; ++n)
            sum += e[i_start + m][j_start + n] * filter[m][n];
    return sum;
}

void pool(double convout[][IMAGESIZE], double poolout[][POOLOUTSIZE])
{
    for( int i = 0; i < POOLOUTSIZE; ++i)
        for( int j = 0; j < POOLOUTSIZE; ++j)
            convout[i][j] = calcpooling(convout, i * POOLSIZE, j * POOLSIZE);
}

double calcpooling(double convout[][IMAGESIZE], int i, int j)
{   // 平均池化
    double sum = 0.0;
    for( int m = 0; m < POOLSIZE; ++m)
        for( int n = 0; n < POOLSIZE; ++n)
            sum += convout[i + m][j + n];
    return sum / POOLSIZE / POOLSIZE;
}