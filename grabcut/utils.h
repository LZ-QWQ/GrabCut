#pragma once
#include "GMM.h"
#include "GrabCut.h"
#include "hist.h"
#include "graph.h"

typedef Graph<double, double, double> GraphCut;

using namespace std;
using namespace cv;

//Mask�ھ��ο��ڵĳ�ʼ��
void initMaskInRect(Mat& _mask, Rect& _rect, const Size& _imgsize);

//��kmeans��ʼ��GMM
void initGMM(const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM);

//����betaֵ
double calcuBeta(const Mat& img);

//ƽ�����Ԥ����
void calcuNWeight(const Mat& _img, Mat& _l, Mat& _ul, Mat& _u, Mat& _ur, double _beta, double _gamma);

//����GMM�еĸ�˹����
void assignGMMS(const Mat& _img, const Mat& _mask, const GMM& _bgdGMM, const GMM& _fgdGMM, Mat& _partIndex);

//ѧϰGMM����~,��ʵ���ǰ�������ȥ���ơ���
void learnGMMs(const Mat& _img, const Mat& _mask, GMM& _bgdGMM, GMM& _fgdGMM, const Mat& _partIndex);

//���������ʽת����ͼ�Ӷ�����max flow/min cut�㷨���
GraphCut* getGraph(const Mat& _img, const Mat& _mask, const GMM& _bgdGMM, const GMM& _fgdGMM, double _lambda, const Mat& _l, const Mat& _ul, const Mat& _u, const Mat& _ur);

//min cut��
void estimateSegmentation(GraphCut* _graph, Mat& _mask);

//Hist ����betaֵ
double calcuBeta_Hist(const Mat& img);

//Hist ƽ�����Ԥ����
void calcuNWeight_Hist(const Mat& _img, Mat& _l, Mat& _ul, Mat& _u, Mat& _ur, double _beta, double _gamma);

//Hist ���������ʽת����ͼ�Ӷ�����max flow/min cut�㷨���
GraphCut* getGraph_Hist(const Mat& _img, const Mat& _mask, const Hist& _bgdHist, const Hist& _fgdHist, double _lambda, const Mat& _l, const Mat& _ul, const Mat& _u, const Mat& _ur);

GraphCut* getGraph_Hist2(const Mat& _img, const Mat& _mask, const Hist& _bgdHist, const Hist& _fgdHist, double _lambda, const Mat& _l, const Mat& _ul, const Mat& _u, const Mat& _ur);

//����ǰ����������������������ɫֱ��ͼʹ��
void cacluCount(const Mat& mask, int& fgdCount, int&bgdCount);

double calcuGMM_energy(const Mat& _img, const Mat& _mask, const GMM& _bgdGMM, const GMM& _fgdGMM, double _lambda, const Mat& _l, const Mat& _ul, const Mat& _u, const Mat& _ur);
