#include <stdio.h>
#include <stdlib.h>

#define SEED 65535      // 随机数SEED
#define GU 0            // 石头
#define CYOKI 1         // 剪刀
#define PA 2            // 布
#define WIN 1           // 胜
#define LOSE -1         // 负
#define DRAW 0          // 平
#define ALPHA 0.01      // 学习速率
#define ACTION_N 3      // 动作数
#define _CRT_SECURE_NO_WARNINGS

/*
在数据文件 j.txt 输入 N a b c
游戏回合数 N 对手出拳比例 石头 a 剪子 b 布 c
*/
int hand(double rate[]);                        // 根据随机数与出拳比例决定出拳 
double frand(void);                             // 随机函数

int main()
{
    int n=0, N, gain;;
    int myhand, ohand;
    double my_rate[ACTION_N] = {1, 1, 1};
    double o_rate[ACTION_N] = {1, 1, 1};
    int count[ACTION_N] = {0, 0, 0};
    int payoffmattrix[ACTION_N][ACTION_N] = {
        {DRAW, WIN, LOSE},
        {LOSE, DRAW, WIN},
        {WIN, LOSE, DRAW}
    };      // 胜负矩阵
    FILE* fp = fopen("j.txt", "r");
    if (fp == NULL)
    {
        printf("error in open file\n");
        return 0;
    }
    fscanf(fp, "%d", &N); 
    printf("Round:%d\no_rate:", N);
    for (int i=0; i<ACTION_N; i++)
    {   
        fscanf(fp, "%lf", o_rate + i); 
        printf("%.2lf ", o_rate[i]);
    }
    printf("\n");
    fclose(fp);
    while (n < N)
    {   
        ohand = hand(o_rate);
        myhand = hand(my_rate);                     // 按照比例出拳
        ++count[ohand];
        gain = payoffmattrix[myhand][ohand];        // 判断胜负
        printf("round_%04d: %d %d %d ", n, myhand, ohand,gain); 
        my_rate[myhand] += gain * ALPHA * my_rate[myhand]; // 学习
        printf("  GU:%.2lf CYOKI:%.2lf PA:%.2lf\n", my_rate[GU], my_rate[CYOKI], my_rate[PA]);
        n++;
    }
    printf("o_hand count: GU:%d CYOKI:%d PA:%d\n", count[GU], count[CYOKI], count[PA]);
    return 0;
}

// 根据随机数与出拳比例决定出拳 
// double rate出拳比例
int hand(double rate[])
{   
    double sum = rate[0];
    for(int i=1; i<ACTION_N; i++)
    {
        sum += rate[i];
    }
    double rand = frand(), cumsum_prob=rate[0] / sum;
    int action = 0;
    while((cumsum_prob < rand) && (action < ACTION_N))
    {
        action++;
        cumsum_prob += rate[action] / sum;
    }
    return action;
}

double frand(void)
{
    return (double)rand() / RAND_MAX;
}