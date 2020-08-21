#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <math.h>
using namespace std;
using namespace cv;
//�洢�����Ϣ
struct point {
	int x, y;
};
struct para_point {
	point p;
	int index, section;
	double delta, sigma;
};
struct inf_point {
	point p;
	int dis, area;
};
//�洢����
struct dands {
	int delta, sigma;
};
//һЩ���͵Ķ���
typedef vector<double[30][10]> Energyfunction;
typedef vector<dands[30][10]> Record;
typedef vector<para_point> Contour;
typedef unordered_map<int, inf_point> Strip;
//��������Ϊ���ڵ�8��
#define nstep 8
const int nx[nstep] = { 0, 1, 0, -1, -1, -1, 1, 1 };
const int ny[nstep] = { 1, 0, -1, 0, -1, 1, -1, 1 };

#define COE 10000
//TU��width�����������е�ʵ�֣�Ϊ6��
#define stripwidth 6
//L��41ָ���Ǳ߳��������L�Ǳ߳���һ��
#define L 20
//ŷʽ����Ϊ1�����ڵ�
#define rstep 4
const int rx[rstep] = { 0,1,0,-1 };
const int ry[rstep] = { 1,0,-1,0 };
#define MAXNUM 9999999;
//���������ķָ����
#define sigmaLevels  10
#define deltaLevels  40
class BorderMatting
{
public:
	BorderMatting();
	~BorderMatting();
	//borderMatting�Ĺ��캯����Ҳ��������ṩ�Ľӿڡ�
	void borderMatting(const Mat& oriImg, const Mat& mask, Mat& borderMask);
private:
	//������������������������в�������
	void ParameterizationContour(const Mat& edge);
	//����������������������������� contour ���й��졣
	void dfs(int x, int y, const Mat& mask, Mat& amask);
	//��ʼ��TU��������ͼ���洢��hash ֵ����������ֵ��
	void StripInit(const Mat& mask);
	//����DP�㷨����������С������ sigma �� delta ��ֵ
	void EnergyMinimization(const Mat& oriImg, const Mat& mask);
	//����ƽ�����ֵ
	inline double varyTerm(double _ddelta, double _dsigma){ return (lamda1*pow(_ddelta, 2.0) + lamda2*pow(_dsigma, 2.0)) / 200; }
	//������ʼ��
	void init(const Mat& img);
	//�����һ����ʼ�㿪ʼ������������������֮��
	double dataTerm(int index, point p, double uf, double ub, double cf, double cb, double delta, double sigma, const Mat& gray);
	//����bfs��������alpha��ֵ
	void CalculateMask(Mat& bordermask, const Mat& mask);
	//��ʾ������ͼƬ
	void display(const Mat& oriImg, const Mat& mask);
	//��ʽ13������lamda��ֵ
	const int lamda1 = 50;
	const int lamda2 = 1000;
	//������������
	int sections; 
	//�������ϲ�ͬ��Ϊ���ĵ�����������������ϵ�ĸ�����
	int areaCount;
	//����
	Contour contour; 
	//TU
	Strip strip; 
	int rows, cols;
	//ʹ��DP�㷨ʱ�洢�м�ֵ
	double ef[5000][deltaLevels][sigmaLevels];
	dands rec[5000][deltaLevels][sigmaLevels];
	vector<dands> vecds;
};
