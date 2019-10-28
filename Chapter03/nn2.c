#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUTNO 2       // 输入层Cell数
#define HIDDENNO 2      // 隐含层Cell数
#define HALF_RAND_MAX (RAND_MAX / 2)
#define ALPHA 10        // 学习系数
#define SEED 65535      // 随机数SEED
#define MAXINPUTNO 50   // 学习数据的最大个数
#define MAX_TRAIN_TIMES 100
#define BIGNUM 100      // 误差初始值
#define LIMIT 0.001     // 误差上限值
#define _CRT_SECURE_NO_WARNINGS

void initwo(double wo[HIDDENNO + 1]);    // 输出层权重初始化
void initwh(double wh[HIDDENNO][INPUTNO + 1]);  // 中间层初始化
int getdata(const char* filename, double e[][INPUTNO + 1]);   // 读取学习数据
double forward(double wh[HIDDENNO][INPUTNO + 1], double wo[HIDDENNO + 1],
               double hi[], double e[INPUTNO]); // 前向计算
void olearn(double wo[HIDDENNO + 1], double hi[], double e[INPUTNO + 1],
            double o);                        // 输出层权重学习
void hlearn(double wh[HIDDENNO][INPUTNO + 1], double wo[HIDDENNO + 1],
            double hi[], double e[INPUTNO + 1], double o);  // 隐含层学习
void print_w(double wh[HIDDENNO][INPUTNO + 1],
             double wo[HIDDENNO + 1]);    // 输出输出层
double drnd(void);                      // 随机数生成 -1~1
double sigmod(double u);                // sigmod激活函数

int main()
{   
    double wh[HIDDENNO][INPUTNO + 1];
    double wo[HIDDENNO + 1]; 
    double e[MAXINPUTNO][INPUTNO + 1];
    double hi[HIDDENNO + 1];
    double o;
    double err = BIGNUM;
    int n_of_e;

    srand(SEED);

    initwh(wh);
    initwo(wo);
    print_w(wh, wo);

    n_of_e = getdata("nn1.txt", e);

    printf("学习数据个数：%d\n", n_of_e);
    int count = 0;
    while(err > LIMIT && count <= MAX_TRAIN_TIMES)
    {
        err = 0.0;
        for(int i=0; i<n_of_e; ++i)
        {
            o = forward(wh, wo, hi, e[i]);
            // 出力層の重みを調整
            olearn(wo, hi, e[i], o);
            // 中間層の重みを調整
            hlearn(wh, wo, hi, e[i], o);
            // ２乗誤差
            err += (o - e[i][INPUTNO]) * (o - e[i][INPUTNO]);
        }
        ++count;
        printf("count: %d\terr: %.4lf\n", count, err);
    }
    print_w(wh, wo);

    for(int i=0; i<n_of_e; ++i)
    {
        printf("%d", i);
        for (int j=0;j<INPUTNO+1; ++j)
            printf("%.2lf", e[i][j]);
        o = forward(wh, wo, hi, e[i]);
        printf("%.2lf\n", o);
    }
    return 0;
}

// 中间层初始化
void initwh(double wh[HIDDENNO][INPUTNO + 1]) 
{
    for (int i = 0; i < HIDDENNO; i++) 
    {
        for (int j = 0; j < INPUTNO + 1; j++) 
        {
            wh[i][j] = drnd();
        }
    }
}

// 输出层初始化
void initwo(double wo[HIDDENNO + 1]) 
{
    for (int i = 0; i < HIDDENNO + 1; i++) 
    {
        wo[i] = drnd();
    }
}

int getdata(const char* filename, double e[][INPUTNO + 1]) 
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
        if (j > INPUTNO) 
        {  
            j = 0;
            n_of_e++;
        }
    }
    fclose(fp);
    return n_of_e;
}

double forward(double wh[HIDDENNO][INPUTNO + 1], double wo[HIDDENNO + 1],
               double hi[], double e[INPUTNO])
{   
    int i, j;     
    double u, o;

    // 中间层计算
    for (i = 0; i < HIDDENNO; i++) 
    {
        u = 0;
        for (j = 0; j < INPUTNO; j++) 
        {
            u += e[j] * wh[i][j];
        }
        u -= wh[i][j];  
        hi[i] = sigmod(u);
    }

    //输出层计算
    o = 0;
    for (i = 0; i < HIDDENNO; i++) {
        o += hi[i] * wo[i];
    }
    o -= wo[i];

    return sigmod(o);
}


void olearn(double wo[HIDDENNO + 1], double hi[], double e[INPUTNO + 1],
            double o) 
{   
    /*
    sigmod  误差微分 o * (1 - o)
    w 更新函数 (y真实 - y估计) * o * (1 - o)
    */
    double d;
    int i;
    d = (e[INPUTNO] - o) * o * (1 - o);
    for (i = 0; i < HIDDENNO; i++) 
    {
        wo[i] += ALPHA * d * hi[i];
    }
    wo[i] += ALPHA * d * (-1.0); 
}

void hlearn(double wh[HIDDENNO][INPUTNO + 1], double wo[HIDDENNO + 1],
            double hi[], double e[INPUTNO + 1], double o) {
    double dj;
    int i, j;
    for (i = 0; i < HIDDENNO; i++) {
        dj = hi[i] * (1 - hi[i]) * wo[i] * (e[INPUTNO] - o) * o * (1 - o);
        for (j = 0; j < INPUTNO; j++) {
            // 中間層への入力毎の重みを更新
            wh[i][j] += ALPHA * dj * e[j];
        }
        wh[i][j] += ALPHA * dj * (-1.0);  // 閾値の学習
    }
}

void print_w(double wh[HIDDENNO][INPUTNO + 1], double wo[HIDDENNO + 1])
{   
    printf("中间层:");
    for (int i = 0; i < HIDDENNO; i++) 
    {
        for (int j = 0; j < INPUTNO; j++) 
        {
            printf("%.2lf ", wh[i][j]);
        }
        printf("\n");
    }
    // 打印输出层
    printf("输出层:");
    for (int i = 0; i < HIDDENNO; i++) 
    {
        printf("%.2lf ", wo[i]);
    }
    printf("\n");
}

double sigmod(double u)
{
    return 1.0 / (1.0+ exp(-u));
}

double drnd(void) 
{
    return (double)(rand() - HALF_RAND_MAX) / HALF_RAND_MAX;
}