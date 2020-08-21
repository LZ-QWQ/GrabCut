#pragma once
#include "GMM.h"
#include "GrabCut.h"
#include "hist.h"
#include "graph.h"

typedef Graph<double, double, double> GraphCut;

using namespace std;
using namespace cv;

//Mask在矩形框内的初始化
void initMaskInRect(Mat& _mask, Rect& _rect, const Size& _imgsize);

//用kmeans初始化GMM
void initGMM(const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM);

//计算beta值
double calcuBeta(const Mat& img);

//平滑项的预计算
void calcuNWeight(const Mat& _img, Mat& _l, Mat& _ul, Mat& _u, Mat& _ur, double _beta, double _gamma);

//分配GMM中的高斯分量
void assignGMMS(const Mat& _img, const Mat& _mask, const GMM& _bgdGMM, const GMM& _fgdGMM, Mat& _partIndex);

//学习GMM参数~,其实就是按照样本去估计。。
void learnGMMs(const Mat& _img, const Mat& _mask, GMM& _bgdGMM, GMM& _fgdGMM, const Mat& _partIndex);

//将能量表达式转换成图从而利用max flow/min cut算法求解
GraphCut* getGraph(const Mat& _img, const Mat& _mask, const GMM& _bgdGMM, const GMM& _fgdGMM, double _lambda, const Mat& _l, const Mat& _ul, const Mat& _u, const Mat& _ur);

//min cut！
void estimateSegmentation(GraphCut* _graph, Mat& _mask);

//Hist 计算beta值
double calcuBeta_Hist(const Mat& img);

//Hist 平滑项的预计算
void calcuNWeight_Hist(const Mat& _img, Mat& _l, Mat& _ul, Mat& _u, Mat& _ur, double _beta, double _gamma);

//Hist 将能量表达式转换成图从而利用max flow/min cut算法求解
GraphCut* getGraph_Hist(const Mat& _img, const Mat& _mask, const Hist& _bgdHist, const Hist& _fgdHist, double _lambda, const Mat& _l, const Mat& _ul, const Mat& _u, const Mat& _ur);

GraphCut* getGraph_Hist2(const Mat& _img, const Mat& _mask, const Hist& _bgdHist, const Hist& _fgdHist, double _lambda, const Mat& _l, const Mat& _ul, const Mat& _u, const Mat& _ur);

//计算前背景各自像素数量，给颜色直方图使用
void cacluCount(const Mat& mask, int& fgdCount, int&bgdCount);

double calcuGMM_energy(const Mat& _img, const Mat& _mask, const GMM& _bgdGMM, const GMM& _fgdGMM, double _lambda, const Mat& _l, const Mat& _ul, const Mat& _u, const Mat& _ur);
