#pragma once
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>

using namespace cv;

//RGB空间的GMM模型，三维高斯分量~
class GMM 
{
public:
	static const int K = 5;//混合高斯模型数量
	GMM(Mat& _model);//构造函数，输入是fgdmodel 或 bgdmodel
	
	
	int choice(const cv::Vec3d) const;//计算一个颜色应该是属于哪个组件（高斯概率最高的项）	
	double possibility(int, const cv::Vec3d) const;//计算某个颜色属于某个组件的可能性（高斯概率）

	//计算数据项权重，全部加起来是为了赋值给图的权重用以判断归属前景还是背景，不过感觉跟论文表达不大一样啊？？
	double tWeight(const cv::Vec3d) const;

	void learningBegin();//学习前初始化	
	void addSample(int _i, const Vec3d _color);//添加单个的点
	void learningEnd();//计算结果

private:
	void calcuInvAndDet(int _i);
	Mat GMM_model;//存储模型~ 保存了个寂寞吧(ˉ▽ˉ；)...
	double *weight, *mean, *cov;//存储权重、均值、协方差
	double covInv[K][3][3];//协方差逆
	double covDet[K];//协方差行列式

	//用于学习过程中保存中间数据的变量 learningBegin→addSample→learningEnd
	double sums[K][3];
	double prods[K][3][3];
	int sampleCounts[K];
	int totalSampleCount;
};