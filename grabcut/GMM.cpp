#include "GMM.h"

using namespace std;

GMM::GMM(Mat& _model)
{
	if (_model.empty())
	{
		_model.create(1, K*13, CV_64FC1);//Ȩ�� ��ֵ Э�����󣨰���չ����
		_model.setTo(Scalar(0));
	}
	GMM_model = _model;
	weight = GMM_model.ptr<double>(0);
	mean = weight + K;
	cov = mean + 3 * K;
	//���ĳ�����Ȩ�ز�Ϊ0���������Э������������ʽ
	for (int i = 0; i < K; i++)
		if (weight[i] > 0)
			calcuInvAndDet(i);
	totalSampleCount = 0;
}

//����һ����ɫӦ���������ĸ��������˹������ߵ��
int GMM::choice(const Vec3d _color) const 
{
	int k = 0;
	double max = 0;
	for (int i = 0; i < K; i++) 
	{
		double p = possibility(i, _color);//Ϊʲô����ô��İ���������Dn��������Ĳ�һ�� opencv...
		if (p > max) 
		{
			k = i;
			max = p;
		}
	}
	return k;
}

//����ĳ����ɫ����ĳ������Ŀ����ԣ���˹���ʣ�
double GMM::possibility(int _i, const Vec3d _color) const 
{
	double res = 0;
	if (weight[_i] > 0) 
	{
		CV_Assert(covDet[_i] > std::numeric_limits<double>::epsilon());
		Vec3d diff = _color;
		double* m = mean + 3 * _i;
		diff[0] -= m[0]; diff[1] -= m[1]; diff[2] -= m[2];//���ǰ��վ���˷�չ������
		double mult = diff[0] * (diff[0] * covInv[_i][0][0] + diff[1] * covInv[_i][1][0] + diff[2] * covInv[_i][2][0])
			+ diff[1] * (diff[0] * covInv[_i][0][1] + diff[1] * covInv[_i][1][1] + diff[2] * covInv[_i][2][1])
			+ diff[2] * (diff[0] * covInv[_i][0][2] + diff[1] * covInv[_i][1][2] + diff[2] * covInv[_i][2][2]);
		res = 1.0f / sqrt(covDet[_i]) * exp(-0.5f*mult);
	}
	return res;
}

//����������Ȩ�أ�ȫ����������Ϊ�˸�ֵ��ͼ��Ȩ�������жϹ���ǰ�����Ǳ����������о������ı�ﲻ��һ��������  opencv...
double GMM::tWeight(const Vec3d _color) const
{
	double res = 0;
	for (int ci = 0; ci < K; ci++)
		res += weight[ci] * possibility(ci, _color);
	return res;
}


void GMM::learningBegin()
{
	//��Ҫ�õ��м������0
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
	//�ı��м������ֵ
	for (int i = 0; i < 3; i++) {
		sums[_i][i] += _color[i];
		for (int j = 0; j < 3; j++)
			prods[_i][i][j] += _color[i] * _color[j];
	}
	sampleCounts[_i]++;
	totalSampleCount++;
}

//������ӵ����ݣ������µĲ������
void GMM::learningEnd() 
{
	//��ʵ�������nά��˹�ļ�����Ȼ����
	const double variance = 0.01;
	for (int i = 0; i < K; i++) 
	{
		int n = sampleCounts[i];
		if (n == 0)	weight[i] = 0;
		else 
		{
			//�����˹ģ���µĲ���
			//Ȩ��
			CV_Assert(totalSampleCount > 0);
			weight[i] = 1.0 * n / totalSampleCount;
			//��ֵ
			double * m = mean + 3 * i;
			for (int j = 0; j < 3; j++) 
			{
				m[j] = sums[i][j] / n;
			}
			//Э����
			double* c = cov + 9 * i;
			for (int p = 0; p < 3; p++) 
			{
				for (int q = 0; q < 3; q++) 
				{
					c[p * 3 + q] = prods[i][p][q] / n - m[p] * m[q];// 
				}
			}
			//��άֱ��չ������
			double dtrm = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
			//�������ʽֵ̫С�������һЩ������ɧ������
			if (dtrm <= std::numeric_limits<double>::epsilon()) 
			{
				c[0] += variance;
				c[4] += variance;
				c[8] += variance;
			}
			//����Э������������ʽ
			calcuInvAndDet(i);
		}
	}
}

//����Э���������������ʽ��ֵ
void GMM::calcuInvAndDet(int _i) 
{
	if (weight[_i] > 0) 
	{
		double *c = cov + 9 * _i;
		//��άֱ�Ӱ���չ������
		double dtrm = covDet[_i] = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
		//����������壬�����������~
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