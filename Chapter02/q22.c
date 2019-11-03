#include <stdio.h>
#include <stdlib.h>

#define SEED 65535      // 随机数SEED
#define QUARTER_RAND_MAX ((RAND_MAX + 1) / 4)
#define GENMAX 100      // 最大训练次数
#define STATE_N 64      // z状态数
#define REWARD  10      // 奖励
#define GOAL    54      // 目标状态
#define UP       0      // 上
#define DOWN     1      // 下
#define LEFT     2      // 左
#define RIGHT    3      // 右
#define ACTION_N 4      // 动作数
#define LEVEL  512      // 单次循环最大次数
#define ALPHA    0.1    // 学习速率
#define GAMMA    0.95   // 折扣系数
#define EPSILON  0.3    // 执行随机策略的概率
#define _CRT_SECURE_NO_WARNINGS


int rand03();
double frand();
void printqvalue(double qvalue[STATE_N][ACTION_N]);
int selecta(int s, double qvalue[STATE_N][ACTION_N]);
double updateq(int s, int s_next, int a, double qvalue[STATE_N][ACTION_N]);
int set_a_by_q(int s, double qvalue[STATE_N][ACTION_N]);
int step(int s, int a);
int TRANSFER_MAT[4] = {-8, 8, -1, 1};


int main()
{
    int s, s_next, t, action;
    double qvalue[STATE_N][ACTION_N];

    srand(SEED);

    for (int i=0; i<STATE_N; ++i)
    {
        for(int j=0; j<ACTION_N;++j)
            qvalue[i][j] = frand();
        if (i<7)
            qvalue[i][UP] = -RAND_MAX;
        if (i>55)
            qvalue[i][DOWN] = -RAND_MAX;
        if (i%8 == 0)
            qvalue[i][LEFT] = -RAND_MAX;
        if (i%8 == 7)
            qvalue[i][RIGHT] = -RAND_MAX;
    }
    printqvalue(qvalue);
    printf("start_run\n");
    for (int i=0; i<GENMAX; ++i)
    {
        s = 0;
        for(t=0; t<LEVEL;)
        {
            action = selecta(s, qvalue);
            if(qvalue[s][action] == -RAND_MAX)      // 无效动作
                continue;
            else
                ++t;
            s_next = step(s, action);
            qvalue[s][action] = updateq(s, s_next, action, qvalue);
            s = s_next;
            if (s == GOAL)
                break;
        }
        printf(">>>>>>>>>>Iter%04d step:%04d<<<<<<<<<<\n", i, t);
        // printqvalue(qvalue);
    }
    printqvalue(qvalue);
    printf("run over\n");
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
        action = rand03();
    else
        action = set_a_by_q(s, qvalue);
    return action;
}

// 根据Q值表来选择行动
int set_a_by_q(int s, double qvalue[STATE_N][ACTION_N])
{
    int best_action = -1;
    double best_q = -RAND_MAX;
    for(int i=0; i<ACTION_N; i++)
    {
        if (qvalue[s][i] > best_q)
        {
            best_action = i;
            best_q = qvalue[s][i];
        }
    }
    return best_action;
}

// 按照action前进一步
int step(int s, int action)
{
    return s + TRANSFER_MAT[action];
}

// 输出Q值表
// qvalue[STATE_N][ACTION_N] Q值表
void printqvalue(double qvalue[STATE_N][ACTION_N])
{   
    char action_str[5] = "UDLR";
    for (int i=0; i<STATE_N; ++i)
    {   
        if (i == GOAL)
        {
            printf("# ");
            continue;
        }
        double best_q = -RAND_MAX;
        int best_action = 0;
        for (int j=0; j<ACTION_N;++j)
        {
            if (qvalue[i][j] > best_q)
            {
                best_action = j;
                best_q = qvalue[i][j];
            }
        }
        printf("%c ", action_str[best_action]);
        if (i % 8 == 7)
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