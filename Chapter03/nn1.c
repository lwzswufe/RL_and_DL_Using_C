#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUTNO 2       // 输入层Cell数
#define HALF_RAND_MAX (RAND_MAX / 2)
#define ALPHA 10        // 学习系数
#define SEED 65535      // 随机数SEED
#define MAXINPUTNO 50   // 学习数据的最大个数
#define MAX_TRAIN_TIMES 100
#define BIGNUM 100      // 误差初始值
#define LIMIT 0.001     // 误差上限值
#define _CRT_SECURE_NO_WARNINGS


void initwo(double wo[INPUTNO + 1]);    // 输出层权重初始化
int getdata(const char* filename, double e[][INPUTNO + 1]);   // 读取学习数据
double forward(double wo[INPUTNO + 1], double e[INPUTNO + 1]);  // 前向计算
void olearn(double wo[INPUTNO + 1], double e[INPUTNO + 1],
            double o);                  // 输出层权重学习
void print_w(double wo[INPUTNO + 1]);   // 输出输出层
double drnd(void);                      // 随机数生成 -1~1
double sigmod(double u);                // sigmod激活函数

int main()
{
    double wo[INPUTNO + 1];
    double e[MAXINPUTNO][INPUTNO + 1];
    double o;
    double err = BIGNUM;
    int n_of_e;

    srand(SEED);

    initwo(wo);
    print_w(wo);

    n_of_e = getdata("nn1.txt", e);

    printf("学习数据个数：%d\n", n_of_e);
    int count = 0;
    while(err > LIMIT && count <= MAX_TRAIN_TIMES)
    {
        err = 0.0;
        for(int j=0; j<n_of_e; ++j)
        {
            o = forward(wo, e[j]);
            olearn(wo, e[j], o);
            err += (o -e[j][INPUTNO]) * (o -e[j][INPUTNO]);
        }
        ++count;
        printf("count: %d\terr: %.4lf\n", count, err);
    }
    print_w(wo);

    for(int i=0; i<n_of_e; ++i)
    {
        printf("%d", i);
        for (int j=0;j<INPUTNO+1; ++j)
            printf("%.2lf", e[i][j]);
        o = forward(wo, e[i]);
        printf("%.2lf\n", o);
    }
    return 0;
}

void initwo(double wo[INPUTNO + 1]) 
{
    for (int i = 0; i < INPUTNO + 1; i++) 
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

double forward(double wo[INPUTNO + 1], double e[INPUTNO + 1]) 
{   
    int i = 0;
    double o = 0;
    for (; i < INPUTNO; ++i) 
    {
        o += e[i] * wo[i];  
    }
    o -= wo[i];  // 阈值处理

    return sigmod(o);
}


void olearn(double wo[INPUTNO + 1], double e[INPUTNO + 1], double o) 
{   
    /*
    sigmod  误差微分 o * (1 - o)
    w 更新函数 (y真实 - y估计) * o * (1 - o)
    */
    double d;
    int i;
    d = (e[INPUTNO] - o) * o * (1 - o);
    for (i = 0; i < INPUTNO; i++) 
    {
        wo[i] += ALPHA * d * e[i];
    }
    wo[i] += ALPHA * d * (-1.0);  // 阈值更新
}

void print_w(double wo[INPUTNO + 1])
{
    // 打印输出层
    printf("输出层:");
    for (int i = 0; i < INPUTNO; i++) {
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