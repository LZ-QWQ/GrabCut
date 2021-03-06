#pragma once
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include<vector>

//Hist 效果感觉有大问题，Lab不收敛，RGB下必须a交互，可能我代码有问题，我无能为力了

//https://github.com/MCG-NKU/CmCode 主要参考自此处

using namespace cv;
using namespace std;

//这两模板摘自CMM老师的代码。。
template<class T, int D> inline T vecSqrDist(const Vec<T, D> &v1, const Vec<T, D> &v2) { T s = 0; for (int i = 0; i < D; i++) s += cv::detail::sqr(v1[i] - v2[i]); return s; } // out of range risk for T = byte, ...
template<class T, int D> inline T    vecDist(const Vec<T, D> &v1, const Vec<T, D> &v2) { return sqrt(vecSqrDist(v1, v2)); } // out of range risk for T = byte, ...

class Hist
{
public:
	static const int bins = 12;//根据程老师论文设置的😓,RGB上量化
	static const int defaultNums[3];//关于量化的设置


	//这两函数只能用一个，有点笨只能出此下策了,返回的是bin图，，
	void learnHist_FGD(const Mat& img3f, const Mat &mask, const int fgdCount,
		double ratio = 0.96, const int clrNums[3] = defaultNums);//ratio 保留颜色的覆盖率？看下论文
	void learnHist_BGD(const Mat& img3f, const Mat &mask, const int bgdCount,
		double ratio = 0.96, const int clrNums[3] = defaultNums);//ratio 保留颜色的覆盖率？看下论文

	double getweight(Vec3f color, const int clrNums[3] = defaultNums) const;
	vector<Vec3i> color3i;//这个是用来保存 量化后的。。数值的
private:	
	Mat _colorNum;//用来保存某种颜色特征的数量
	Mat _weight;//有colorNum计算得出的权重~
	int maxNum;//高频出现的颜色总数，覆盖率>=ratio
	
};