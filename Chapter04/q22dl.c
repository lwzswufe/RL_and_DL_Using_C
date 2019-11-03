#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define SEED 65535      // 随机数SEED
#define QUARTER_RAND_MAX ((RAND_MAX) / 4 + 1)
#define HALF_RAND_MAX (RAND_MAX / 2)
#define GENMAX 100      // 最大训练次数
#define STATE_N 64      // 状态数
#define REWARD  10      // 奖励
#define GOAL    54      // 目标状态
#define UP       0      // 上
#define DOWN     1      // 下
#define LEFT     2      // 左
#define RIGHT    3      // 右
#define ACTION_N 4      // 动作数
#define LEVEL  512      // 单次循环最大次数
#define ALPHA    0.05   // 学习速率
#define GAMMA    0.95   // 折扣系数
#define EPSILON  0.3    // 执行随机策略的概率
// 卷积神经网络
#define IMAGESIZE 8     // 输入图像边长
#define F_SIZE 3        // 卷积过滤器的大小
#define F_NO 2          // 卷积过滤器的数量
#define POOLOUTSIZE 3   // 池化层大小
#define POOLSIZE 2      // 池化区域大小 注意  POOLSIZE * POOLOUTSIZE = IMAGESIZE + 1 - F_SIZE
// 神经网络
#define INPUTNO (POOLOUTSIZE * POOLOUTSIZE * F_NO)        // 输入层Cell数
#define HIDDENNO 6      // 隐含层Cell数
#define OUTPUTNO 4      // 输出层Cell数
#define NNALPHA 1       // 学习系数
#define _CRT_SECURE_NO_WARNINGS

/*
gcc q22dl.c -o q22dl.exe
gcc q22dl.c -lm -o q22dl.o
*/
struct Net
{
    double wh[HIDDENNO][INPUTNO + 1];
    double wo[OUTPUTNO][HIDDENNO + 1]; 
    double hi[HIDDENNO + 1];
    double o[OUTPUTNO];
    double filter[F_NO][F_SIZE][F_SIZE];
    int path[LEVEL];
    int step_n;
};
typedef struct Net Net; 

// 强化学习函数
int rand03();
double frand();
void printqvalue(Net* net_ptr);
int selecta(int s, Net* net_ptr);
double updateq(int s, int s_next, int a, Net* net_ptr);
int set_a_by_q(int s, Net* net_ptr);
int step(int s, int a);
double calcqvalue(Net* net, double e[INPUTNO], int s, int action);
// 卷积神经网络函数
void init_filter(double filter[F_NO][F_SIZE][F_SIZE]);
void conv(double filter[][F_SIZE], double e[][IMAGESIZE], double convout[][IMAGESIZE]);     // 卷积函数
double calcconv(double filter[][F_SIZE], double e[][IMAGESIZE], int i, int j);
void pool(double convout[][IMAGESIZE], double poolout[][POOLOUTSIZE]);
double calcpooling(double convout[][IMAGESIZE], int i, int j);
void set_e_by_s(int s, double filter[F_NO][F_SIZE][F_SIZE], double e[INPUTNO + OUTPUTNO]);
// 全连接神经网络参数
void initwo(double wo[OUTPUTNO][HIDDENNO + 1]); // 输出层权重初始化
void initwh(double wh[HIDDENNO][INPUTNO + 1]);  // 中间层初始化
void forward(Net* net_ptr, double e[INPUTNO], int action);  // 前向计算
void olearn(Net* net_ptr, double e[INPUTNO + OUTPUTNO], int action);  // 输出层权重学习
void hlearn(Net* net_ptr, double e[INPUTNO + OUTPUTNO], int action);  // 隐含层学习
void print_w(Net* net_ptr);
double drnd(void);                      // 随机数生成 -1~1
double sigmod(double u);                // sigmod激活函数




int main()
{
    int s, s_next, t, action;
    struct Net net;
    double e[INPUTNO + OUTPUTNO];
    srand(SEED);
    init_filter(net.filter);
    initwh(net.wh);
    initwo(net.wo);
    print_w(&net);
    // 初始化完毕

    printf("start_run\n");
    for (int i=0; i<GENMAX; ++i)
    {
        s = 0;
        memset(net.path, -1, sizeof(int) * LEVEL);
        net.step_n = 0;
        if(i >= 50)
        {
            int a = 1;
        }
        for(; net.step_n<LEVEL; ++net.step_n)
        {   
            action = selecta(s, &net);
            s_next = step(s, action);
            net.path[net.step_n] = s_next; 
            // 更新Q值
            set_e_by_s(s, net.filter, e);
            e[INPUTNO + action] = updateq(s, s_next, action, &net);
            // 
            forward(&net, e, action);
            olearn(&net, e, action);
            hlearn(&net, e, action);
            //
            // printf("step:%d, %02d->%02d action:%d\n", t, s, s_next, action);
            s = s_next;
            if (s == GOAL)
                break;
        }
        printf(">>>>>>>>>>Iter%04d step:%04d<<<<<<<<<<\n", i, net.step_n);
        printqvalue(&net);
    }
    printf("run over\n");
    printqvalue(&net);
    return 0;
}

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>强化学习相关函数>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
double calcqvalue(Net* net, double e[INPUTNO], int s, int action)
{   
    if (step(s, action) == GOAL)
        return REWARD;
    switch(action)
    {   
        case UP:
            if(s<=7)
                return 0;
            break;
        case DOWN:
            if(s>=56)
                return 0;
            break;
        case LEFT:
            if(s % 8 == 0)
                return 0;
            break;
        case RIGHT:
            if(s % 8 == 7)
                return 0;
            break;
        default:
            return 0.0;
    }
    forward(net, e, action);
    return net->o[action];
}
// 更新Q值表
// qvalue[STATE_N][ACTION_N] Q值表
// int s      当前状态
// int s_next 下一状态
// int action 动作
double updateq(int s, int s_next, int a, Net* net)
{
    double q_now, q_next, qv;
    double e[INPUTNO + OUTPUTNO] = {0};
    // 获取当前e
    set_e_by_s(s, net->filter, e);
    // 获取本期 以及下一期Q值
    q_now =  calcqvalue(net, e, s, a);
    q_next = calcqvalue(net, e, s_next, a);
    // 更新Q值
    if (s_next == GOAL)
        qv = q_now + ALPHA * ( REWARD - q_now );
    else
        qv = q_now + ALPHA *(GAMMA * q_next - q_now);
    return qv;
}

// 选择行动
int selecta(int s, Net* net)
{
    int action, is_rand;
    double e[INPUTNO + OUTPUTNO] = {0};
    // 获取当前e
    set_e_by_s(s, net->filter, e);
    if (frand() < EPSILON)
    {
        do
        {
            action = rand03();
        } 
        while (calcqvalue(net, e, s, action) <= 0);
        is_rand = 1;
    }
    else
    {
        action = set_a_by_q(s, net);
        is_rand = 0;
    }
    return action;
}
// 获取下一步的最优Q值
double get_q_next(int s, int a, Net* net)
{
    int best_action = -1;
    double best_q = -RAND_MAX, q_arr[4];
    double e[INPUTNO + OUTPUTNO] = {0};
    int s_next = step(s, a);
    // 如果s无法采取行动a
    if (s == s_next)
        return 0;
    if (s == GOAL)
        return REWARD;
    set_e_by_s(s, net->filter, e);
    for (int a_next=0; a_next<ACTION_N; a_next++)
    {
        double q = calcqvalue(net, e, s_next, a_next);
        if (q > best_q)
            best_q = q;
    }
    return best_q * GAMMA;
}

// 根据Q值表来选择行动
int set_a_by_q(int s, Net* net)
{
    int best_action = -1;
    double best_q = -RAND_MAX, q_arr[4], q;
    double e[INPUTNO + OUTPUTNO] = {0};
    set_e_by_s(s, net->filter, e);
    for(int a=0; a<ACTION_N; a++)
    {   
        q = calcqvalue(net, e, s, a);
        // q = get_q_next(s, a, net);
        q_arr[a] = q;
        if (q > best_q)
        {
            best_action = a;
            best_q = q;
        }
    }
    // printf("state:%d  ", s);
    // for (int a=0; a<ACTION_N; a++)
    // {  
    //     printf("%.2lf, ", q_arr[a]);
    // }
    // printf("\n");
    int transfer[4] = {8, -8, -1, 1};
    int s_next = s + transfer[best_action];
    if (s_next < 0 || s_next >= STATE_N)
    {
        int a = 1;
    }
    return best_action;
}

// 按照action前进一步
int step(int s, int action)
{   
    int transfer[4] = {-8, 8, -1, 1};
    int s_next = s + transfer[action];
    if (s_next < 0 || s_next >= STATE_N)
    {
        return s;
    }
    return s_next;
}

// 输出Q值表
// qvalue[STATE_N][ACTION_N] Q值表
void printqvalue(Net* net)
{   
    char action_str[5] = "UDLR";
    double e[INPUTNO + OUTPUTNO] = {0};
    double q_value[STATE_N][ACTION_N];
    for (int s=0; s<STATE_N; ++s)
    {   
        if (s == GOAL)
        {
            printf("# ");
            continue;
        }
        double best_q = 0, q;
        int best_action = 0;
        set_e_by_s(s, net->filter, e);
        for (int a=0; a<ACTION_N;++a)
        {   
            q = calcqvalue(net, e, s, a);
            q_value[s][a] = q;
            if (q > best_q)
            {
                best_action = a;
                best_q = q;
            }
        }
        printf("%c ", action_str[best_action]);
        if (s % 8 == 7)
            printf("\n");
    }
    for (int s=0; s<STATE_N; ++s)
    {
        for( int a=0; a<ACTION_N; ++a)
            printf("%.2lf ", q_value[s][a]);
        printf("\n");
    }
}

// 生成0-1的随机数
double frand(void)
{
    return (double)rand() / RAND_MAX;
}


// 随机选择0 1 2 3
int rand03()
{   
    return rand() / QUARTER_RAND_MAX;
}

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>神经网络相关函数>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


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

void forward(Net* net, double e[INPUTNO], int action)
{   
    // 中间层计算
    for (int i = 0; i < HIDDENNO; i++) 
    {
        double u = 0;
        for (int j = 0; j < INPUTNO; j++) 
        {
            u += e[j] * net->wh[i][j];
        }
        u -= net->wh[i][INPUTNO];  
        net->hi[i] = sigmod(u);
    }

    // 输出层计算
    net->o[action] = 0;
    for (int j = 0; j < HIDDENNO; ++j) 
    {
        net->o[action] += net->hi[j] * net->wo[action][j];
    }
    net->o[action] -= net->wo[action][HIDDENNO];
    net->o[action] = sigmod(net->o[action]);
}


void olearn(Net* net, double e[INPUTNO + OUTPUTNO], int action)
{   
    /*
    sigmod  误差微分 net->o * (1 - net->o)
    w 更新函数 (y真实 - y估计) * net->o * (1 - net->o)
    */
    double d = (e[INPUTNO + action] - net->o[action]) * net->o[action] * (1 - net->o[action]);
    for (int j = 0; j < HIDDENNO; ++j) 
    {
        net->wo[action][j] += ALPHA * d * net->hi[j];
    }
    // beta项的学习
    net->wo[action][HIDDENNO] += ALPHA * d * (-1.0);
}

void hlearn(Net* net, double e[INPUTNO + OUTPUTNO], int action) 
{
    for (int j = 0; j < HIDDENNO; ++j)
    {   
        double dj = net->hi[j] * (1 - net->hi[j]) * net->wo[action][j] * (e[INPUTNO + action] - net->o[action]) * net->o[action] * (1 - net->o[action]);
        for (int k = 0; k < INPUTNO; ++k)
        {
            net->wh[j][k] += ALPHA * dj * e[k];
        }
        // beta项的学习
        net->wh[j][INPUTNO] += ALPHA * dj * (-1.0);
    }
}

void print_w(Net* net)
{   
    printf("hidden layers:\n");
    for (int i = 0; i < HIDDENNO; i++) 
    {
        for (int j = 0; j < INPUTNO + 1; j++) 
        {
            printf("%.2lf ", net->wh[i][j]);
        }
        printf("\n");
    }
    // 打印输出层
    printf("output layers:\n");
    for (int i = 0; i < OUTPUTNO; i++) 
    {   
        for (int j = 0; j < HIDDENNO + 1; j++)
        {
            printf("%.2lf ", net->wo[i][j]);
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
            poolout[i][j] = calcpooling(convout, i * POOLSIZE, j * POOLSIZE);
}

double calcpooling(double convout[][IMAGESIZE], int i, int j)
{   // 平均池化
    double sum = 0.0, mean = 0.0;
    for( int m = 0; m < POOLSIZE; ++m)
        for( int n = 0; n < POOLSIZE; ++n)
            sum += convout[i + m][j + n];
    mean = sum / POOLSIZE / POOLSIZE;
    return mean;
}

void init_filter(double filter[F_NO][F_SIZE][F_SIZE])
{
    for ( int i = 0; i < F_NO; ++i)
        for ( int j = 0; j < F_SIZE; ++j)
            for  ( int k = 0; k < F_SIZE; ++k)
                filter[i][j][k] = drnd();
}

void set_e_by_s(int s, double filter[F_NO][F_SIZE][F_SIZE], double e[INPUTNO + OUTPUTNO])
{
    double image[IMAGESIZE][IMAGESIZE];
    double convout[IMAGESIZE][IMAGESIZE];
    double poolout[POOLOUTSIZE][POOLOUTSIZE];

    // 卷积输入
    memset(image, 0, sizeof(double) * IMAGESIZE * IMAGESIZE);
    memset(convout, 0, sizeof(double) * IMAGESIZE * IMAGESIZE);
    memset(poolout, 0, sizeof(double) * POOLOUTSIZE * POOLOUTSIZE);
    image[s % IMAGESIZE][s / IMAGESIZE] = 1;
    // 生成全连接层输入数据
    for (int i = 0; i < F_NO; ++i)
    {   
        // 卷积
        conv(filter[i], image, convout);
        // 池化
        pool(convout, poolout);
        int pool_st = i * POOLOUTSIZE * POOLOUTSIZE;
        for ( int j = 0; j < POOLOUTSIZE; ++j)
            for  ( int k = 0; k < POOLOUTSIZE; ++k)
                e[pool_st + POOLOUTSIZE * j + k] = poolout[j][k];
    }
    // 清空教师数据
    memset(e + INPUTNO, 0, sizeof(double) * OUTPUTNO);
}