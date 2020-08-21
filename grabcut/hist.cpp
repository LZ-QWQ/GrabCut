#include "hist.h"
#include <map>

const int Hist::defaultNums[3] = { 12,12,12 };

void Hist::learnHist_FGD(const Mat& img3f, const Mat &mask, const int fgdCount ,double ratio, const int clrNums[3])
{
	//减一个这么小的浮点数是为什么
	float clrTmp[3] = { clrNums[0] - 0.0001f, clrNums[1] - 0.0001f, clrNums[2] - 0.0001f };
	int w[3] = { clrNums[1] * clrNums[2], clrNums[2], 1 };

	CV_Assert(img3f.data != NULL);
	Mat idx1i = Mat::zeros(img3f.size(), CV_32S);
	int rows = img3f.rows, cols = img3f.cols;
	if (img3f.isContinuous() && idx1i.isContinuous() && mask.isContinuous())//哦豁，，如果连续就可以通过这样子直接接上了。。
	{
		cols *= rows;
		rows = 1;
	}

	// Build color pallet
	map<int, int> pallet;
	for (int y = 0; y < rows; y++)
	{
		const float* imgData = img3f.ptr<float>(y);
		const uchar* maskptr = mask.ptr<uchar>(y);
		int* idx = idx1i.ptr<int>(y);
		for (int x = 0; x < cols; x++, imgData += 3)
		{
			if (maskptr[x] == GC_FGD || maskptr[x] == GC_PR_FGD)
			{
				//其实就是将颜色分布映射到0~(12^3-1)即0~1727，所以才有上面的-0.0001f
				idx[x] = (int)(imgData[0] * clrTmp[0])*w[0] + (int)(imgData[1] * clrTmp[1])*w[1] + (int)(imgData[2] * clrTmp[2]);
				pallet[idx[x]] ++;
			}
		}
	}

	// Find significant colors
	maxNum = 0;
	{
		int count = 0;
		vector<pair<int, int>> num; // (num, color) pairs in num
		num.reserve(pallet.size());
		for (map<int, int>::iterator it = pallet.begin(); it != pallet.end(); it++)
			num.push_back(pair<int, int>(it->second, it->first)); // (color, num) pairs in pallet
		sort(num.begin(), num.end(), std::greater<pair<int, int>>());//从大到小排序

		maxNum = (int)num.size();
		if (maxNum == 0)
		{
			cout << "。。FGD_Hist为空" << endl;
			return;
		}
		//int maxDropNum = cvRound(rows * cols * (1 - ratio));//四舍五入,丢掉的那些颜色！
		int maxDropNum = cvRound(fgdCount * (1 - ratio));//四舍五入,丢掉的那些颜色！
		for (int crnt = num[maxNum - 1].first; crnt < maxDropNum && maxNum > 1; maxNum--)
			crnt += num[maxNum - 2].first;

		//以下两个是用来保证丢弃后的maxNum不过大也不过小
		maxNum = min(maxNum, 256); // To avoid very rarely case
		if (maxNum <= 10)
			maxNum = min(10, (int)num.size());

		pallet.clear();
		for (int i = 0; i < maxNum; i++)
			pallet[num[i].second] = i;//依占比顺序？

		color3i.clear();
		color3i.reserve(num.size());
		for (unsigned int i = 0; i < num.size(); i++)
		{
			//这里都是整除把，或者不是整除也被变整了，就分到了三个维度0~11各类
			color3i[i][0] = num[i].second / w[0];
			color3i[i][1] = num[i].second % w[0] / w[1];
			color3i[i][2] = num[i].second % w[1];
		}

		//把被丢弃颜色的像素点根据距离最近原则进行分配，这里距离计算用的是RGB空间耶
		for (unsigned int i = maxNum; i < num.size(); i++)
		{
			int simIdx = 0, simVal = INT_MAX;
			for (int j = 0; j < maxNum; j++)
			{
				int d_ij = vecSqrDist<int, 3>(color3i[i], color3i[j]);
				if (d_ij < simVal)
					simVal = d_ij, simIdx = j;
			}
			pallet[num[i].second] = pallet[num[simIdx].second];
		}
	}

	Mat _color3f = Mat::zeros(1, maxNum, CV_32FC3);

	_colorNum.create(_color3f.size(), CV_32S);

	Vec3f* color = (Vec3f*)(_color3f.data);
	int* colorNum = (int*)(_colorNum.data);
	int totalNum = 0;
	for (int y = 0; y < rows; y++)
	{
		const Vec3f* imgData = img3f.ptr<Vec3f>(y);
		const uchar* maskptr = mask.ptr<uchar>(y);
		int* idx = idx1i.ptr<int>(y);
		for (int x = 0; x < cols; x++)
		{
			if (maskptr[x] == GC_FGD || maskptr[x] == GC_PR_FGD)
			{
				idx[x] = pallet[idx[x]];//改成丢弃后的分配序号
				color[idx[x]] += imgData[x];
				colorNum[idx[x]] ++;
				totalNum++;
			}
		}
	}
	_weight.create(_color3f.size(), CV_64F);
	double* weightptr = (double*)(_weight.data);

	cout << "FGD权重:" << endl;
	double temp_total = 0;
	for (int i = 0; i < _color3f.cols; i++)
	{
		color[i] /= (float)colorNum[i];
		weightptr[i] = (double)colorNum[i] / totalNum;
		cout << color3i[i] << "    " << weightptr[i] << endl;
		temp_total += weightptr[i];
	}
	//cout << temp_total << endl;
	
	
}

void Hist::learnHist_BGD(const Mat& img3f, const Mat &mask, const int bgdCount,double ratio, const int clrNums[3])
{
	//减一个这么小的浮点数是为什么
	float clrTmp[3] = { clrNums[0] - 0.0001f, clrNums[1] - 0.0001f, clrNums[2] - 0.0001f };
	int w[3] = { clrNums[1] * clrNums[2], clrNums[2], 1 };

	CV_Assert(img3f.data != NULL);
	Mat idx1i = Mat::zeros(img3f.size(), CV_32S);
	int rows = img3f.rows, cols = img3f.cols;
	if (img3f.isContinuous() && idx1i.isContinuous() && mask.isContinuous())//哦豁，，如果连续就可以通过这样子直接接上了。。
	{
		cols *= rows;
		rows = 1;
	}

	// Build color pallet
	map<int, int> pallet;
	for (int y = 0; y < rows; y++)
	{
		const float* imgData = img3f.ptr<float>(y);
		const uchar* maskptr = mask.ptr<uchar>(y);
		int* idx = idx1i.ptr<int>(y);
		for (int x = 0; x < cols; x++, imgData += 3)
		{
			if (maskptr[x] == GC_BGD || maskptr[x] == GC_PR_BGD)
			{
				//其实就是将颜色分布映射到0~(12^3-1)即0~1727，所以才有上面的-0.0001f
				idx[x] = (int)(imgData[0] * clrTmp[0])*w[0] + (int)(imgData[1] * clrTmp[1])*w[1] + (int)(imgData[2] * clrTmp[2]);
				pallet[idx[x]] ++;
			}
		}
	}

	// Find significant colors
	maxNum = 0;
	{
		int count = 0;
		vector<pair<int, int>> num; // (num, color) pairs in num
		num.reserve(pallet.size());
		for (map<int, int>::iterator it = pallet.begin(); it != pallet.end(); it++)
			num.push_back(pair<int, int>(it->second, it->first)); // (color, num) pairs in pallet
		sort(num.begin(), num.end(), std::greater<pair<int, int>>());//从大到小排序

		maxNum = (int)num.size();
		//int maxDropNum = cvRound(rows * cols * (1 - ratio));//四舍五入,丢掉的那些颜色！
		int maxDropNum = cvRound(bgdCount * (1 - ratio));//四舍五入,丢掉的那些颜色！
		for (int crnt = num[maxNum - 1].first; crnt < maxDropNum && maxNum > 1; maxNum--)
			crnt += num[maxNum - 2].first;

		//以下两个是用来保证丢弃后的maxNum不过大也不过小
		maxNum = min(maxNum, 256); // To avoid very rarely case
		if (maxNum <= 10)
			maxNum = min(10, (int)num.size());

		pallet.clear();
		for (int i = 0; i < maxNum; i++)
			pallet[num[i].second] = i;//依占比顺序？

		color3i.clear();
		color3i.reserve(num.size());
		for (unsigned int i = 0; i < num.size(); i++)
		{
			//这里都是整除把，或者不是整除也被变整了，就分到了三个维度0~11各类
			color3i[i][0] = num[i].second / w[0];
			color3i[i][1] = num[i].second % w[0] / w[1];
			color3i[i][2] = num[i].second % w[1];
		}

		//把被丢弃颜色的像素点根据距离最近原则进行分配，这里距离计算用的是RGB空间耶
		for (unsigned int i = maxNum; i < num.size(); i++)
		{
			int simIdx = 0, simVal = INT_MAX;
			for (int j = 0; j < maxNum; j++)
			{
				int d_ij = vecSqrDist<int, 3>(color3i[i], color3i[j]);
				if (d_ij < simVal)
					simVal = d_ij, simIdx = j;
			}
			pallet[num[i].second] = pallet[num[simIdx].second];
		}
	}
	
	Mat _color3f = Mat::zeros(1, maxNum, CV_32FC3);

	
	_colorNum.create(_color3f.size(), CV_32S);


	Vec3f* color = (Vec3f*)(_color3f.data);
	int* colorNum = (int*)(_colorNum.data);
	int totalNum = 0;
	for (int y = 0; y < rows; y++)
	{
		const Vec3f* imgData = img3f.ptr<Vec3f>(y);
		const uchar* maskptr = mask.ptr<uchar>(y);
		int* idx = idx1i.ptr<int>(y);
		for (int x = 0; x < cols; x++)
		{
			if (maskptr[x] == GC_BGD || maskptr[x] == GC_PR_BGD)
			{
				idx[x] = pallet[idx[x]];//改成丢弃后的分配序号
				color[idx[x]] += imgData[x];
				colorNum[idx[x]] ++;
				totalNum++;
			}
		}
	}
	_weight.create(_color3f.size(), CV_64F);
	double* weightptr = (double*)(_weight.data);

	cout << "BGD权重:" << endl;
	double temp_total = 0;
	for (int i = 0; i < _color3f.cols; i++)
	{
		color[i] /= (float)colorNum[i];
		weightptr[i] = (double)colorNum[i] / totalNum;
		cout << color3i[i] << "    " << weightptr[i] << endl;
		temp_total += weightptr[i];
	}
	//cout << temp_total << endl;
}

double Hist::getweight(Vec3f color,const int clrNums[3]) const
{
	float clrTmp[3] = { clrNums[0] - 0.0001f, clrNums[1] - 0.0001f, clrNums[2] - 0.0001f };
	int w[3] = { clrNums[1] * clrNums[2], clrNums[2], 1 };

	int temp=(int)(color[0] * clrTmp[0])*w[0] + (int)(color[1] * clrTmp[1])*w[1] + (int)(color[2] * clrTmp[2]);

	Vec3i color3i_temp;
	color3i_temp[0] = temp / w[0];
	color3i_temp[1] = temp % w[0] / w[1];
	color3i_temp[2] = temp % w[1];

	int simIdx = 0, simVal = INT_MAX;
	for (int j = 0; j < maxNum; j++)
	{
		int d_ij = vecSqrDist<int, 3>(color3i_temp, color3i[j]);
		if (d_ij < simVal)
		{
			simVal = d_ij;
			simIdx = j;
		}
	}
	if (maxNum == 0)return 1;//出去以后有个log 先这样
	double* weightptr = (double*)(_weight.data);
	double temp_weight = weightptr[simIdx];
	return temp_weight;
}