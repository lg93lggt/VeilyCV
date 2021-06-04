#include <cuda_runtime.h> 
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

# define DATA_NUM 200   // 数据量(分辨率) 200x200x200
# define DISTANCE 0.002 // 步长
# define MASK_NUM 14    // 掩膜数量
# define THRESHOLD 12   // 严格程度阈值
# define MASK_WIDTH 1080 // 掩膜宽
# define MASK_HEIGHT 1440 // 掩膜高

__global__ void Projection(float* c,int* mask, int* result_voxels)
{
    // 获取全局索引
    int index1 = threadIdx.x + blockIdx.x * blockDim.x;
    int index2 = threadIdx.y + blockIdx.y * blockDim.y;
    int index3 = threadIdx.z + blockIdx.z * blockDim.z;
    
    // 数据初始化
    float eps = 1e-10;
    float pt[4] = {DISTANCE*index1, DISTANCE*index2, DISTANCE*index3, 1}; 
    int voxels[MASK_NUM] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 1,1};

    for (int index=0;index<MASK_NUM;index++){
        double _p[3] = {0, 0, 0};
        // _p = c * pt 投影
        for (int q=0;q<3;q++){
            for (int w=0;w<4;w++){
                _p[q] += c[index*3*4+q*4+w] * pt[w];
            }
        }
        // _p = _p / (_p[2]+eps);
        _p[0] = _p[0] / (_p[2]+eps);
        _p[1] = _p[1] / (_p[2]+eps);

        // 四舍五入
        int col = (int)(_p[0]+0.5);
        int row = (int)(_p[1]+0.5);

        // 因为row和col有越界情况，越界的全置0
        if (row<MASK_HEIGHT && col<MASK_WIDTH){
            if (mask[index*MASK_HEIGHT*MASK_WIDTH+row*MASK_WIDTH+col]==0){
                voxels[index]=0;
            }
        }else{
            voxels[index]=0;
        }
        // 将MASK_NUM个voxels对应位置相加得到result_voxels
        result_voxels[index1*DATA_NUM*DATA_NUM+index2*DATA_NUM+index3] += voxels[index];
    }
    // 非严格判断，当 threshold>12时置1，<12全部置0
    if (result_voxels[index1*DATA_NUM*DATA_NUM+index2*DATA_NUM+index3]>THRESHOLD){
        // printf("**到这里了**");
        result_voxels[index1*DATA_NUM*DATA_NUM+index2*DATA_NUM+index3] = 1;
    }else{
        result_voxels[index1*DATA_NUM*DATA_NUM+index2*DATA_NUM+index3] = 0;
    }
}


int main()
{
    int start0;
    int end0;
    start0 = clock();
    int mask_num = MASK_WIDTH*MASK_HEIGHT*MASK_NUM;
    int voxels_num = DATA_NUM*DATA_NUM*DATA_NUM;
    int c_num = MASK_NUM*3*4;

    int MaskBytes =  mask_num* sizeof(int);
    int VoxelBytes = voxels_num* sizeof(int);
    int cBytes = c_num* sizeof(float);

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

    // 申请host内存
    int *mask;
    int *result_voxels;
    float *c;
    mask = (int*)malloc(MaskBytes);
    result_voxels = (int*)malloc(VoxelBytes);
    c = (float*)malloc(cBytes);

    // 初始化数据
    // c = intrinsic * extrinsic
    for (int q=0;q<MASK_NUM;q++){
        for (int i=0;i<3;i++){
            for (int j = 0; j < 4; j++){
                for (int k = 0; k < 3; k++){
                    c[q*3*4+i*4+j] += intrinsic[i][k] * extrinsic[q][k][j];
                }
            }
        }
    }
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

    // 生成result_voxels
    for (int i=0;i<DATA_NUM;i++){
        for (int j=0;j<DATA_NUM;j++){
            for (int k=0;k<DATA_NUM;k++){
                result_voxels[i*DATA_NUM*DATA_NUM+j*DATA_NUM+k] = 0;
            }
        }
    }

    // 申请device内存
    int *d_mask;
    int *d_voxels;
    float *d_c;
    cudaMalloc((void**)&d_mask, MaskBytes);
    cudaMalloc((void**)&d_voxels, VoxelBytes);
    cudaMalloc((void**)&d_c, cBytes);

    // 将host数据拷贝到device
    int start2;
    int end2;
    start2 = clock();
    cudaMemcpy((void*)d_mask, (void*)mask, MaskBytes, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_voxels, (void*)result_voxels, VoxelBytes, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_c, (void*)c, cBytes, cudaMemcpyHostToDevice);
    end2 = clock();
    printf("数据从host to device耗时%fs",double(end2-start2)/CLOCKS_PER_SEC);
    printf("\n");

    // 定义kernel的执行配置
    dim3 blockSize(10,10,10);
    dim3 gridSize(20,20,20);
    
    int start3;
    int end3;
    start3 = clock();
    // 执行kernel  float* c,int* pt, int* _p, int* mask, int* result_voxels
    Projection<<<gridSize,blockSize>>>(d_c, d_mask,d_voxels);
    end3 = clock();
    printf("运行Kernel耗时%fs",double(end3-start3)/CLOCKS_PER_SEC);
    printf("\n");


    int start1;
    int end1;
    start1 = clock();
    // 将device得到的结果拷贝到host
    cudaMemcpy((void*)result_voxels, (void*)d_voxels, VoxelBytes, cudaMemcpyDeviceToHost);
    end1 = clock();
    printf("数据从device to host耗时%fs",double(end1-start1)/CLOCKS_PER_SEC);
    printf("\n");
    
    // 将结果保存到txt文档 以便比对
    FILE *fp;
    fp=fopen("./VoxelsResult001.txt","w");//打开文件以便写入数据
    for (int i=0;i<DATA_NUM;i++){
        fprintf(fp,"\n");
        for (int j=0;j<DATA_NUM;j++){
            for (int k=0;k<DATA_NUM;k++){
                fprintf(fp,"%d ",result_voxels[i*DATA_NUM*DATA_NUM+j*DATA_NUM+k]);
            }
        }
    }
    fclose(fp); //写入完毕，关闭文件

    // 释放device内存
    cudaFree(d_mask);
    cudaFree(d_voxels);
    cudaFree(d_c);
    // 释放host内存
    free(mask);
    free(result_voxels);
    free(c);

    end0 = clock();
    printf("整个代码耗时%fs",double(end0-start0)/CLOCKS_PER_SEC);
    printf("\n");
    return 0;
}