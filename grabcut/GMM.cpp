#include "GMM.h"

using namespace std;

GMM::GMM(Mat& _model)
{
	if (_model.empty())
	{
		_model.create(1, K*13, CV_64FC1);//权重 均值 协方差阵（按行展开）
		_model.setTo(Scalar(0));
	}
	GMM_model = _model;
	weight = GMM_model.ptr<double>(0);
	mean = weight + K;
	cov = mean + 3 * K;
	//如果某个项的权重不为0，则计算其协方差的逆和行列式
	for (int i = 0; i < K; i++)
		if (weight[i] > 0)
			calcuInvAndDet(i);
	totalSampleCount = 0;
}

//计算一个颜色应该是属于哪个组件（高斯概率最高的项）
int GMM::choice(const Vec3d _color) const 
{
	int k = 0;
	double max = 0;
	for (int i = 0; i < K; i++) 
	{
		double p = possibility(i, _color);//为什么是这么算的啊？？不是Dn嘛！！跟论文不一样 opencv...
		if (p > max) 
		{
			k = i;
			max = p;
		}
	}
	return k;
}

//计算某个颜色属于某个组件的可能性（高斯概率）
double GMM::possibility(int _i, const Vec3d _color) const 
{
	double res = 0;
	if (weight[_i] > 0) 
	{
		CV_Assert(covDet[_i] > std::numeric_limits<double>::epsilon());
		Vec3d diff = _color;
		double* m = mean + 3 * _i;
		diff[0] -= m[0]; diff[1] -= m[1]; diff[2] -= m[2];//就是按照矩阵乘法展开计算
		double mult = diff[0] * (diff[0] * covInv[_i][0][0] + diff[1] * covInv[_i][1][0] + diff[2] * covInv[_i][2][0])
			+ diff[1] * (diff[0] * covInv[_i][0][1] + diff[1] * covInv[_i][1][1] + diff[2] * covInv[_i][2][1])
			+ diff[2] * (diff[0] * covInv[_i][0][2] + diff[1] * covInv[_i][1][2] + diff[2] * covInv[_i][2][2]);
		res = 1.0f / sqrt(covDet[_i]) * exp(-0.5f*mult);
	}
	return res;
}

//计算数据项权重，全部加起来是为了赋值给图的权重用以判断归属前景还是背景，不过感觉跟论文表达不大一样啊？？  opencv...
double GMM::tWeight(const Vec3d _color) const
{
	double res = 0;
	for (int ci = 0; ci < K; ci++)
		res += weight[ci] * possibility(ci, _color);
	return res;
}


void GMM::learningBegin()
{
	//对要用的中间变量赋0
	for (int i = 0; i < K; i++) {
		for (int j = 0; j < 3; j++)
			sums[i][j] = 0;
		for (int p = 0; p < 3; p++) {
			for (int q = 0; q < 3; q++) {
				prods[i][p][q] = 0;
			}
		}
		sampleCounts[i] = 0;
	}
	totalSampleCount = 0;
}

void GMM::addSample(int _i, const Vec3d _color)
{
	//改变中间变量的值
	for (int i = 0; i < 3; i++) {
		sums[_i][i] += _color[i];
		for (int j = 0; j < 3; j++)
			prods[_i][i][j] += _color[i] * _color[j];
	}
	sampleCounts[_i]++;
	totalSampleCount++;
}

//根据添加的数据，计算新的参数结果
void GMM::learningEnd() 
{
	//其实这里就是n维高斯的极大似然估计
	const double variance = 0.01;
	for (int i = 0; i < K; i++) 
	{
		int n = sampleCounts[i];
		if (n == 0)	weight[i] = 0;
		else 
		{
			//计算高斯模型新的参数
			//权重
			CV_Assert(totalSampleCount > 0);
			weight[i] = 1.0 * n / totalSampleCount;
			//均值
			double * m = mean + 3 * i;
			for (int j = 0; j < 3; j++) 
			{
				m[j] = sums[i][j] / n;
			}
			//协方差
			double* c = cov + 9 * i;
			for (int p = 0; p < 3; p++) 
			{
				for (int q = 0; q < 3; q++) 
				{
					c[p * 3 + q] = prods[i][p][q] / n - m[p] * m[q];// 
				}
			}
			//三维直接展开计算
			double dtrm = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
			//如果行列式值太小，则加入一些噪音，骚操作啊
			if (dtrm <= std::numeric_limits<double>::epsilon()) 
			{
				c[0] += variance;
				c[4] += variance;
				c[8] += variance;
			}
			//计算协方差的逆和行列式
			calcuInvAndDet(i);
		}
	}
}

//计算协方差矩阵的逆和行列式的值
void GMM::calcuInvAndDet(int _i) 
{
	if (weight[_i] > 0) 
	{
		double *c = cov + 9 * _i;
		//三维直接按列展开计算
		double dtrm = covDet[_i] = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
		//利用逆矩阵定义，伴随矩阵来算~
		covInv[_i][0][0] = (c[4] * c[8] - c[5] * c[7]) / dtrm;
		covInv[_i][1][0] = -(c[3] * c[8] - c[5] * c[6]) / dtrm;
		covInv[_i][2][0] = (c[3] * c[7] - c[4] * c[6]) / dtrm;
		covInv[_i][0][1] = -(c[1] * c[8] - c[2] * c[7]) / dtrm;
		covInv[_i][1][1] = (c[0] * c[8] - c[2] * c[6]) / dtrm;
		covInv[_i][2][1] = -(c[0] * c[7] - c[1] * c[6]) / dtrm;
		covInv[_i][0][2] = (c[1] * c[5] - c[2] * c[4]) / dtrm;
		covInv[_i][1][2] = -(c[0] * c[5] - c[2] * c[3]) / dtrm;
		covInv[_i][2][2] = (c[0] * c[4] - c[1] * c[3]) / dtrm;
	}
}