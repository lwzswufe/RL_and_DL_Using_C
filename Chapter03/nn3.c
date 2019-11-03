#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INPUTNO 2       // 输入层Cell数
#define HIDDENNO 2      // 隐含层Cell数
#define OUTPUTNO 3      // 输出层Cell数
#define HALF_RAND_MAX (RAND_MAX / 2)
#define ALPHA 10        // 学习系数
#define SEED 65535      // 随机数SEED
#define MAXINPUTNO 50   // 学习数据的最大个数
#define MAX_TRAIN_TIMES 500
#define BIGNUM 100      // 误差初始值
#define LIMIT 0.001     // 误差上限值
#define _CRT_SECURE_NO_WARNINGS

void initwo(double wo[OUTPUTNO][HIDDENNO + 1]);    // 输出层权重初始化
void initwh(double wh[HIDDENNO][INPUTNO + 1]);  // 中间层初始化
int getdata(const char* filename, double e[][INPUTNO + OUTPUTNO]);   // 读取学习数据
void forward(double wh[HIDDENNO][INPUTNO + 1], double wo[OUTPUTNO][HIDDENNO + 1],
               double hi[HIDDENNO + 1], double e[INPUTNO], double o[OUTPUTNO]); // 前向计算
void olearn(double wo[OUTPUTNO][HIDDENNO + 1], double hi[HIDDENNO + 1], double e[INPUTNO + OUTPUTNO],
            double o[OUTPUTNO]);                        // 输出层权重学习
void hlearn(double wh[HIDDENNO][INPUTNO + 1], double wo[OUTPUTNO][HIDDENNO + 1],
            double hi[HIDDENNO + 1], double e[INPUTNO + OUTPUTNO], double o[OUTPUTNO]);  // 隐含层学习
void print_w(double wh[HIDDENNO][INPUTNO + 1],
             double wo[OUTPUTNO][HIDDENNO + 1]);    // 输出输出层
double drnd(void);                      // 随机数生成 -1~1
double sigmod(double u);                // sigmod激活函数

int main()
{   
    double wh[HIDDENNO][INPUTNO + 1];
    double wo[OUTPUTNO][HIDDENNO + 1]; 
    double e[MAXINPUTNO][INPUTNO + OUTPUTNO];
    double hi[HIDDENNO + 1];
    double o[OUTPUTNO];
    double err = BIGNUM;
    int n_of_e;

    srand(SEED);

    initwh(wh);
    initwo(wo);
    print_w(wh, wo);
    n_of_e = getdata("nn3.txt", e);

    printf("train data num:%d\n", n_of_e);
    int count = 0;
    while(err > LIMIT && count <= MAX_TRAIN_TIMES)
    {
        err = 0.0;
        for(int i=0; i<n_of_e; ++i)
        {
            forward(wh, wo, hi, e[i], o);
            olearn(wo, hi, e[i], o);
            hlearn(wh, wo, hi, e[i], o);
            for (int j = 0; j < OUTPUTNO; ++j)
                err += (o[j] - e[i][INPUTNO + j]) * (o[j] - e[i][INPUTNO + j]);
        }
        ++count;
        printf("count: %d\terr: %.4lf\n", count, err);
    }
    print_w(wh, wo);

    for(int i = 0; i < n_of_e; ++i)
    {
        printf("data:%d:  ", i);
        for (int j = 0; j < INPUTNO + OUTPUTNO; ++j)
            printf("%.2lf ", e[i][j]);
        forward(wh, wo, hi, e[i], o);
        for (int j = 0; j < OUTPUTNO; ++j)
            printf("%.2lf ", o[j]);
        printf("\n");
    }
    return 0;
}

// 中间层初始化
void initwh(double wh[HIDDENNO][INPUTNO + 1]) 
{
    for (int i = 0; i < HIDDENNO; i++) 
    {
        for (int j = 0; j < INPUTNO + OUTPUTNO; j++) 
        {
            wh[i][j] = drnd();
        }
    }
}

// 输出层初始化
void initwo(double wo[OUTPUTNO][HIDDENNO + 1]) 
{
    for (int i = 0; i < OUTPUTNO; ++i) 
    {   
        for (int j = 0; j < HIDDENNO + 1; ++j)
        {
            wo[i][j] = drnd();
        }
    }
}

int getdata(const char* filename, double e[][INPUTNO + OUTPUTNO]) 
{
    int j = 0;       
    int n_of_e = 0;  
    FILE* fp = fopen(filename, "r");
    if (fp == NULL)
    {
        printf("error in open file:%s\n", filename);
    }
    while (fscanf(fp, "%lf", &e[n_of_e][j]) != EOF) 
    {
        j++;
        if (j >= INPUTNO + OUTPUTNO) 
        {  
            j = 0;
            n_of_e++;
        }
    }
    fclose(fp);
    return n_of_e;
}

void forward(double wh[HIDDENNO][INPUTNO + 1], double wo[OUTPUTNO][HIDDENNO + 1],
               double hi[HIDDENNO + 1], double e[INPUTNO], double o[OUTPUTNO])
{   
    // 中间层计算
    for (int i = 0; i < HIDDENNO; i++) 
    {
        double u = 0;
        for (int j = 0; j < INPUTNO; j++) 
        {
            u += e[j] * wh[i][j];
        }
        u -= wh[i][INPUTNO];  
        hi[i] = sigmod(u);
    }

    //输出层计算
    memset(o, 0, sizeof(double) * OUTPUTNO);
    for (int i = 0; i < OUTPUTNO; ++i)
    {
        for (int j = 0; j < HIDDENNO; ++j) 
        {
            o[i] += hi[j] * wo[i][j];
        }
        o[i] -= wo[i][HIDDENNO];
        o[i] = sigmod(o[i]);
    }
}


void olearn(double wo[OUTPUTNO][HIDDENNO + 1], double hi[HIDDENNO + 1], double e[INPUTNO + OUTPUTNO],
            double o[OUTPUTNO])
{   
    /*
    sigmod  误差微分 o * (1 - o)
    w 更新函数 (y真实 - y估计) * o * (1 - o)
    */
    for (int i = 0; i < OUTPUTNO; ++i)
    {
        double d = (e[INPUTNO + i] - o[i]) * o[i] * (1 - o[i]);
        for (int j = 0; j < HIDDENNO; ++j) 
        {
            wo[i][j] += ALPHA * d * hi[j];
        }
        // beta项的学习
        wo[i][HIDDENNO] += ALPHA * d * (-1.0);
    }
}

void hlearn(double wh[HIDDENNO][INPUTNO + 1], double wo[OUTPUTNO][HIDDENNO + 1],
            double hi[HIDDENNO + 1], double e[INPUTNO + OUTPUTNO], double o[OUTPUTNO]) 
{
    for (int i = 0; i < OUTPUTNO; ++i)
    {
        for (int j = 0; j < HIDDENNO; ++j)
        {   
            double dj = hi[j] * (1 - hi[j]) * wo[i][j] * (e[INPUTNO + i] - o[i]) * o[i] * (1 - o[i]);
            for (int k = 0; k < INPUTNO; ++k)
            {
                wh[j][k] += ALPHA * dj * e[k];
            }
            // beta项的学习
            wh[j][INPUTNO] += ALPHA * dj * (-1.0);
        }
    }
}

void print_w(double wh[HIDDENNO][INPUTNO + 1], double wo[OUTPUTNO][HIDDENNO + 1])
{   
    printf("hidden layers:\n");
    for (int i = 0; i < HIDDENNO; i++) 
    {
        for (int j = 0; j < INPUTNO + 1; j++) 
        {
            printf("%.2lf ", wh[i][j]);
        }
        printf("\n");
    }
    // 打印输出层
    printf("output layers:\n");
    for (int i = 0; i < OUTPUTNO; i++) 
    {   
        for (int j = 0; j < HIDDENNO + 1; j++)
        {
            printf("%.2lf ", wo[i][j]);
        }
        printf("\n");
    }
    
}

double sigmod(double u)
{
    return 1.0 / (1.0+ exp(-u));
}

double drnd(void) 
{
    return (double)(rand() - HALF_RAND_MAX) / HALF_RAND_MAX;
}