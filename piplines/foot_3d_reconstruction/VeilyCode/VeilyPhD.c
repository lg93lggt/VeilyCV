#include<stdio.h>
#include <stdlib.h>
#include <time.h>
# define DATA_NUM 100
# define DISTANCE 0.004
# define MASK_NUM 14
# define THRESHOLD 12
# define MASK_WIDTH 1080
# define MASK_HEIGHT 1440

int main()
{
    // 数据初始化
	float c[MASK_NUM][3][4]={0};
    int num=0;
    int *voxels;
    int *mask;
    int *result_voxels;
    voxels = (int *)malloc(MASK_NUM*DATA_NUM*DATA_NUM*DATA_NUM*sizeof(int));
    mask = (int *)malloc(MASK_NUM*MASK_HEIGHT*MASK_WIDTH*sizeof(int));
    result_voxels = (int *)malloc(DATA_NUM*DATA_NUM*DATA_NUM*sizeof(int));
    float eps = 1e-10;

    // 读取intrinsic
    FILE *fin0 = fopen("./data/intrinsic.txt", "r");
    float intrinsic[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++){
            fscanf(fin0, "%e", &intrinsic[i][j]);
        }
    }
    fclose(fin0);

    // 读取extrinsic， result1.txt为多个extrinsic.txt合并而成
    FILE *fin1 = fopen("./data/result1.txt", "r");
    float extrinsic[MASK_NUM][3][4];
    for (int k = 0; k < MASK_NUM; k++){
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++){
                fscanf(fin1, "%e", &extrinsic[k][i][j]);
            }
        }
    }
    fclose(fin1);

    // 读取mask， result2.txt为多个mask.txt合并而成
    FILE *ff1 = fopen("./data/result2.txt", "r");
    for (int i = 0; i < MASK_NUM; i++) {
        for (int j = 0; j < MASK_HEIGHT; j++){
            for (int k = 0; k < MASK_WIDTH; k++){
                fscanf(ff1, "%d", &mask[i*MASK_HEIGHT*MASK_WIDTH+j*MASK_WIDTH+k]);
            }
        }
    }
    fclose(ff1);
    
    // 生成voxels
    for (int q=0;q<MASK_NUM;q++){
        for (int i=0;i<DATA_NUM;i++){
            for (int j=0;j<DATA_NUM;j++){
                for (int k=0;k<DATA_NUM;k++){
                    voxels[q*DATA_NUM*DATA_NUM*DATA_NUM+i*DATA_NUM*DATA_NUM+j*DATA_NUM+k] = 1;
            }
        }
    }
    }
    // 生成result_voxels
    for (int i=0;i<DATA_NUM;i++){
            for (int j=0;j<DATA_NUM;j++){
                for (int k=0;k<DATA_NUM;k++){
                    result_voxels[i*DATA_NUM*DATA_NUM+j*DATA_NUM+k] = 0;
            }
        }
    }

    // c = intrinsic * extrinsic  投影中间值
    for (int q=0;q<MASK_NUM;q++){
        for (int i = 0; i < 3; i++){
		    for (int j = 0; j < 4; j++){
			    for (int k = 0; k < 3; k++){
                    c[q][i][j] += intrinsic[i][k] * extrinsic[q][k][j];
                }
		    }
	    }
    }

    
    for (int i=0;i<DATA_NUM;i++){
        for (int j=0;j<DATA_NUM;j++){
            for (int k=0;k<DATA_NUM;k++){
                float pt[MASK_NUM][4] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                                    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
                float _p[MASK_NUM][3] = {0};
                // 数据点的生成
                pt[0][0] = DISTANCE*i;pt[0][1] = DISTANCE*j;pt[0][2] = DISTANCE*k;
                pt[1][0] = DISTANCE*i;pt[1][1] = DISTANCE*j;pt[1][2] = DISTANCE*k;
                pt[2][0] = DISTANCE*i;pt[2][1] = DISTANCE*j;pt[2][2] = DISTANCE*k;
                pt[3][0] = DISTANCE*i;pt[3][1] = DISTANCE*j;pt[3][2] = DISTANCE*k;
                pt[4][0] = DISTANCE*i;pt[4][1] = DISTANCE*j;pt[4][2] = DISTANCE*k;
                pt[5][0] = DISTANCE*i;pt[5][1] = DISTANCE*j;pt[5][2] = DISTANCE*k;
                pt[6][0] = DISTANCE*i;pt[6][1] = DISTANCE*j;pt[6][2] = DISTANCE*k;
                pt[7][0] = DISTANCE*i;pt[7][1] = DISTANCE*j;pt[7][2] = DISTANCE*k;
                pt[8][0] = DISTANCE*i;pt[8][1] = DISTANCE*j;pt[8][2] = DISTANCE*k;
                pt[9][0] = DISTANCE*i;pt[9][1] = DISTANCE*j;pt[9][2] = DISTANCE*k;
                pt[10][0] = DISTANCE*i;pt[10][1] = DISTANCE*j;pt[10][2] = DISTANCE*k;
                pt[11][0] = DISTANCE*i;pt[11][1] = DISTANCE*j;pt[11][2] = DISTANCE*k;
                pt[12][0] = DISTANCE*i;pt[12][1] = DISTANCE*j;pt[12][2] = DISTANCE*k;
                pt[13][0] = DISTANCE*i;pt[13][1] = DISTANCE*j;pt[13][2] = DISTANCE*k;

                for(int index = 0 ; index<MASK_NUM; index++){
                    // _p = c * pt 投影
                    for (int q=0;q<3;q++){
                        for (int w=0;w<4;w++){
                            _p[index][q] += c[index][q][w] * pt[index][w];
                        }
                    }
                    // _p = _p / (_p[2]+eps);
                    _p[index][0] = _p[index][0] / (_p[index][2]+eps);
                    _p[index][1] = _p[index][1] / (_p[index][2]+eps);
                    _p[index][2] = _p[index][2] / (_p[index][2]+eps);

                    // 四舍五入
                    int col = (int)(_p[index][0]+0.5);
                    int row = (int)(_p[index][1]+0.5);

                    // 因为row和col有越界情况，越界的全置0
                    if (row<MASK_HEIGHT && col<MASK_WIDTH){
                        if(mask[index*MASK_HEIGHT*MASK_WIDTH+row*MASK_WIDTH+col]==0){
                            voxels[index*DATA_NUM*DATA_NUM*DATA_NUM+i*DATA_NUM*DATA_NUM+j*DATA_NUM+k]=0;
                        }
                    }else{
                        voxels[index*DATA_NUM*DATA_NUM*DATA_NUM+i*DATA_NUM*DATA_NUM+j*DATA_NUM+k]=0;
                    }
                    // 将14个voxels对应位置相加得到result_voxels
                    result_voxels[i*DATA_NUM*DATA_NUM+j*DATA_NUM+k] += voxels[index*DATA_NUM*DATA_NUM*DATA_NUM+i*DATA_NUM*DATA_NUM+j*DATA_NUM+k];
                }

                // 非严格判断，当 threshold>12时置1，<12全部置0
                if (result_voxels[i*DATA_NUM*DATA_NUM+j*DATA_NUM+k]>THRESHOLD){
                    result_voxels[i*DATA_NUM*DATA_NUM+j*DATA_NUM+k] = 1;
                }else{
                    result_voxels[i*DATA_NUM*DATA_NUM+j*DATA_NUM+k] = 0;
                }
                // 先求和与python比对，再单个元素比对
                num += result_voxels[i*DATA_NUM*DATA_NUM+j*DATA_NUM+k];
            }
        }
    }

    // 保存结果以便对比
    FILE *fp;
    fp=fopen("./voc.txt","w");//打开文件以便写入数据
    for (int i=0;i<DATA_NUM;i++){
        fprintf(fp,"\n");
        for (int j=0;j<DATA_NUM;j++){
            for (int k=0;k<DATA_NUM;k++){
                fprintf(fp,"%d\t",result_voxels[i*DATA_NUM*DATA_NUM+j*DATA_NUM+k]);
            }
        }
    }
    fclose(fp); //写入完毕，关闭文件

    // 释放分配的内存
    free(voxels);
    free(mask);
    free(result_voxels);
    return 0;

}