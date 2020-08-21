#pragma once
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>

using namespace cv;

//RGB�ռ��GMMģ�ͣ���ά��˹����~
class GMM 
{
public:
	static const int K = 5;//��ϸ�˹ģ������
	GMM(Mat& _model);//���캯����������fgdmodel �� bgdmodel
	
	
	int choice(const cv::Vec3d) const;//����һ����ɫӦ���������ĸ��������˹������ߵ��	
	double possibility(int, const cv::Vec3d) const;//����ĳ����ɫ����ĳ������Ŀ����ԣ���˹���ʣ�

	//����������Ȩ�أ�ȫ����������Ϊ�˸�ֵ��ͼ��Ȩ�������жϹ���ǰ�����Ǳ����������о������ı�ﲻ��һ��������
	double tWeight(const cv::Vec3d) const;

	void learningBegin();//ѧϰǰ��ʼ��	
	void addSample(int _i, const Vec3d _color);//��ӵ����ĵ�
	void learningEnd();//������

private:
	void calcuInvAndDet(int _i);
	Mat GMM_model;//�洢ģ��~ �����˸���į��(��������)...
	double *weight, *mean, *cov;//�洢Ȩ�ء���ֵ��Э����
	double covInv[K][3][3];//Э������
	double covDet[K];//Э��������ʽ

	//����ѧϰ�����б����м����ݵı��� learningBegin��addSample��learningEnd
	double sums[K][3];
	double prods[K][3][3];
	int sampleCounts[K];
	int totalSampleCount;
};