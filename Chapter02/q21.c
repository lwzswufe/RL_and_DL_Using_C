#include <stdio.h>
#include <stdlib.h>

#define SEED 65535      // 随机数SEED
#define HALF_RAND_MAX (RAND_MAX / 2)
#define GENMAX  50      // 最大训练次数
#define STATE_N  7      // z状态数
#define REWARD  10      // 奖励
#define GOAL     6      // 目标状态
#define UP       0      // 上
#define DOWN     1      // 下
#define ACTION_N 2      // 动作数
#define LEVEL    2
#define ALPHA    0.1    // 学习速率
#define GAMMA    0.9    // 折扣系数
#define EPSILON  0.3    // 执行随机策略的概率
#define _CRT_SECURE_NO_WARNINGS

/* 
从0开始向右移动 每次移动可以选择 右上 或者 右下 移动到6 回合结束
        3
    1       
        4
0
        5
    2
        6
*/
int rand0or1();
double frand();
void printqvalue(double qvalue[STATE_N][ACTION_N]);
int selecta(int s, double qvalue[STATE_N][ACTION_N]);
double updateq(int s, int s_next, int a, double qvalue[STATE_N][ACTION_N]);
int set_a_by_q(int s, double qvalue[STATE_N][ACTION_N]);
int step(int s, int a);

int main()
{
    int s, s_next, t, action;
    double qvalue[STATE_N][ACTION_N];

    srand(SEED);

    for (int i=0; i<STATE_N; ++i)
        for(int j=0; j<ACTION_N;++j)
            qvalue[i][j] = frand();
    printqvalue(qvalue);

    for (int i=0; i<GENMAX; ++i)
    {
        s = 0;
        for(int t=0; t<LEVEL; ++t)
        {
            action = selecta(s, qvalue);
            s_next = step(s, action);
            qvalue[s][action] = updateq(s, s_next, action, qvalue);
            s = s_next;
        }
        printf(">>>>>>>>>>Iter%04d<<<<<<<<<<\n", i);
        printqvalue(qvalue);
    }
    return 0;
}

// 更新Q值表
// qvalue[STATE_N][ACTION_N] Q值表
// int s      当前状态
// int s_next 下一状态
// int action 动作
double updateq(int s, int s_next, int action, double qvalue[STATE_N][ACTION_N])
{
    double qv;
    if (s_next == GOAL)
        qv = qvalue[s][action] + ALPHA * ( REWARD - qvalue[s][action] );
    else
    {   
        int action_next = set_a_by_q(s_next, qvalue);
        qv = qvalue[s][action] + ALPHA *(GAMMA * qvalue[s_next][action_next] - qvalue[s][action]);
    }
    return qv;
}

// 选择行动
int selecta(int s, double qvalue[STATE_N][ACTION_N])
{
    int action;
    if (frand() < EPSILON)
        action = rand0or1();
    else
        action = set_a_by_q(s, qvalue);
    return action;
}

// 根据Q值表来选择行动
int set_a_by_q(int s, double qvalue[STATE_N][ACTION_N])
{
    if(qvalue[s][UP] > qvalue[s][DOWN])
        return UP;
    else
        return DOWN;
}

// 按照action前进一步
int step(int s, int action)
{
    return s * 2 + 1 + action;
}


// 输出Q值表
// qvalue[STATE_N][ACTION_N] Q值表
void printqvalue(double qvalue[STATE_N][ACTION_N])
{   
    // for(int j=0; j<ACTION_N;++j)
    // {
    //     for (int i=0; i<STATE_N; ++i)
    //         printf("%.2f, ", qvalue[i][j]);
    //     printf("\n");
    // }
    printf("\t\t%.2f\n%.2f\t%.2f\n%.2f\t%.2f\n\t\t%.2f\n", 
            qvalue[1][UP], qvalue[0][UP], qvalue[1][DOWN],
            qvalue[0][DOWN], qvalue[2][UP], qvalue[2][DOWN]);
}

// 生成0-1的随机数
double frand(void)
{
    return (double)rand() / RAND_MAX;
}


// 随机选择0或1
int rand0or1()
{
    if (rand() > HALF_RAND_MAX)
        return 1;
    else
        return 0;
}