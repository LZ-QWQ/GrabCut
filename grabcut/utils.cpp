#include "utils.h"

//Mask在矩形框内的初始化
void initMaskInRect(Mat& _mask, Rect& _rect, const Size& _imgsize)
{
	assert(!_mask.empty());
	_mask.setTo(GC_BGD);   //GC_BGD == 0
	_rect.x = max(0, _rect.x);
	_rect.y = max(0, _rect.y);
	_rect.width = min(_rect.width, _imgsize.width - _rect.x);
	_rect.height = min(_rect.height, _imgsize.height - _rect.y);
	(_mask(_rect)).setTo(Scalar(GC_PR_FGD));    //GC_PR_FGD == 3 
}

//用kmeans初始化GMM
void initGMM(const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM)
{
	const int kmeansItCount = 10;
	Mat bgdLabel, fgdLabel;
	vector<Vec3f> bgdSamples, fgdSamples;
	Point p;
	for (p.y = 0; p.y < img.rows; p.y++)
	{
		for (p.x = 0; p.x < img.cols; p.x++)
		{
			if (mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD)
				bgdSamples.push_back((Vec3f)img.at<Vec3b>(p));
			else
				fgdSamples.push_back((Vec3f)img.at<Vec3b>(p));
		}
	}
	Mat _bgdSamples((int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0]);//这个构造函数是为啥呢，查不到 抄！
	kmeans(_bgdSamples, GMM::K, bgdLabel,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, kmeansItCount, 0.0), 3, KMEANS_PP_CENTERS);
	Mat _fgdSamples((int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0]);
	kmeans(_fgdSamples, GMM::K, fgdLabel,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, kmeansItCount, 0.0), 3, KMEANS_PP_CENTERS);

	bgdGMM.learningBegin();
	for (int i = 0; i < (int)bgdSamples.size(); i++)
		bgdGMM.addSample(bgdLabel.at<int>(i, 0), bgdSamples[i]);
	bgdGMM.learningEnd();

	fgdGMM.learningBegin();
	for (int i = 0; i < (int)fgdSamples.size(); i++)
		fgdGMM.addSample(fgdLabel.at<int>(i, 0), fgdSamples[i]);
	fgdGMM.learningEnd();

	return;
}

//β=(2*<(Zm-Zn)^2)^0.5 八邻域邻近的计算~
double calcuBeta(const Mat& _img)
{
	double beta;
	double totalDiff = 0;
	for (int y = 0; y < _img.rows; y++) {
		for (int x = 0; x < _img.cols; x++) {
			Vec3d color = (Vec3d)_img.at<Vec3b>(y, x);
			if (x > 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y, x - 1);
				totalDiff += diff.dot(diff);
			}
			if (y > 0 && x > 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y - 1, x - 1);
				totalDiff += diff.dot(diff);
			}
			if (y > 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y - 1, x);
				totalDiff += diff.dot(diff);
			}
			if (y > 0 && x < _img.cols - 1) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y - 1, x + 1);
				totalDiff += diff.dot(diff);
			}
		}
	}
	if (totalDiff <= std::numeric_limits<double>::epsilon()) beta = 0;//避免太小呢
	else beta = 1.0 / (2 * totalDiff / (8 * _img.cols*_img.rows - 6 * _img.cols - 6 * _img.rows + 4));//计算期望
	return beta;
}

//平滑项V的预计算
void calcuNWeight(const Mat& _img, Mat& _l, Mat& _ul, Mat& _u, Mat& _ur, double _beta, double _gamma) {
	const double gammaDiv = _gamma / std::sqrt(2.0f);// 原论文这里是没有考虑距离倒数了啊？难道这里有问题？
	_l.create(_img.size(), CV_64FC1);
	_ul.create(_img.size(), CV_64FC1);
	_u.create(_img.size(), CV_64FC1);
	_ur.create(_img.size(), CV_64FC1);
	for (int y = 0; y < _img.rows; y++) {
		for (int x = 0; x < _img.cols; x++) {
			Vec3d color = (Vec3d)_img.at<Vec3b>(y, x);
			if (x - 1 >= 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y, x - 1);
				_l.at<double>(y, x) = _gamma * exp(-_beta * diff.dot(diff));
			}
			else _l.at<double>(y, x) = 0;
			if (x - 1 >= 0 && y - 1 >= 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y - 1, x - 1);
				_ul.at<double>(y, x) = gammaDiv * exp(-_beta * diff.dot(diff));
				//_ul.at<double>(y, x) =_gamma * exp(-_beta * diff.dot(diff));

			}
			else _ul.at<double>(y, x) = 0;
			if (y - 1 >= 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y - 1, x);
				_u.at<double>(y, x) = _gamma * exp(-_beta * diff.dot(diff));
			}
			else _u.at<double>(y, x) = 0;
			if (x + 1 < _img.cols && y - 1 >= 0) {
				Vec3d diff = color - (Vec3d)_img.at<Vec3b>(y - 1, x + 1);
				_ur.at<double>(y, x) = gammaDiv * exp(-_beta * diff.dot(diff));
				//_ur.at<double>(y, x) = _gamma * exp(-_beta * diff.dot(diff));
			}
			else _ur.at<double>(y, x) = 0;
		}
	}
}

//分配GMM中的高斯分量
void assignGMMS(const Mat& _img, const Mat& _mask, const GMM& _bgdGMM, const GMM& _fgdGMM, Mat& _partIndex) {
	Point p;//其实我记得论文里不是 for each n in Tu
	for (p.y = 0; p.y < _img.rows; p.y++) {
		for (p.x = 0; p.x < _img.cols; p.x++) {
			Vec3d color = (Vec3d)_img.at<Vec3b>(p);
			uchar t = _mask.at<uchar>(p);
			if (t == GC_BGD || t == GC_PR_BGD)_partIndex.at<int>(p) = _bgdGMM.choice(color);
			else _partIndex.at<int>(p) = _fgdGMM.choice(color);
		}
	}
}

//学习GMM参数~,其实就是按照样本去估计。。
void learnGMMs(const Mat& _img, const Mat& _mask, GMM& _bgdGMM, GMM& _fgdGMM, const Mat& _partIndex)
{
	_bgdGMM.learningBegin();
	_fgdGMM.learningBegin();
	Point p;
	for (int i = 0; i < GMM::K; i++) 
	{
		for (p.y = 0; p.y < _img.rows; p.y++) {
			for (p.x = 0; p.x < _img.cols; p.x++) {
				int tmp = _partIndex.at<int>(p);
				if (tmp == i) {
					if (_mask.at<uchar>(p) == GC_BGD || _mask.at<uchar>(p) == GC_PR_BGD)
						_bgdGMM.addSample(tmp, _img.at<Vec3b>(p));
					else
						_fgdGMM.addSample(tmp, _img.at<Vec3b>(p));
				}
			}
		}
	}
	_bgdGMM.learningEnd();
	_fgdGMM.learningEnd();
}

//将能量表达式转换成图从而利用max flow/min cut算法求解
GraphCut* getGraph(const Mat& _img, const Mat& _mask, const GMM& _bgdGMM, const GMM& _fgdGMM, double _lambda, const Mat& _l, const Mat& _ul, const Mat& _u, const Mat& _ur)
{
	int V_Count=_img.cols*_img.rows;//点集数量
	int E_Count = 2 * (4 * V_Count - 3 * _img.cols - 3 * _img.rows + 2);//边集数量
	GraphCut* _graph = new GraphCut(V_Count, E_Count);
	Point p;
	for (p.y = 0; p.y < _img.rows; p.y++) 
	{
		for (p.x = 0; p.x < _img.cols; p.x++) 
		{
			int vNum = _graph->add_node();
			Vec3b color = _img.at<Vec3b>(p);
			double wSource = 0, wSink = 0;
			if (_mask.at<uchar>(p) == GC_PR_BGD || _mask.at<uchar>(p) == GC_PR_FGD) 
			{
				wSource = -log(_bgdGMM.tWeight(color));
				wSink = -log(_fgdGMM.tWeight(color));
			}
			else if (_mask.at<uchar>(p) == GC_BGD) wSink = _lambda;
			else wSource = _lambda;
			_graph->add_tweights(vNum, wSource, wSink);
			if (p.x > 0) 
			{
				double w = _l.at<double>(p);
				_graph->add_edge(vNum, vNum - 1, w, w);
			}
			if (p.x > 0 && p.y > 0) 
			{
				double w = _ul.at<double>(p);
				_graph->add_edge(vNum, vNum - _img.cols - 1, w, w);
			}
			if (p.y > 0) 
			{
				double w = _u.at<double>(p);
				_graph->add_edge(vNum, vNum - _img.cols, w, w);
			}
			if (p.x < _img.cols - 1 && p.y > 0) 
			{
				double w = _ur.at<double>(p);
				_graph->add_edge(vNum, vNum - _img.cols + 1, w, w);
			}
		}
	}
	return _graph;
}

//min cut！
void estimateSegmentation(GraphCut* _graph, Mat& _mask) 
{
	_graph->maxflow();
	Point p;
	for (p.y = 0; p.y < _mask.rows; p.y++) 
	{
		for (p.x = 0; p.x < _mask.cols; p.x++) 
		{
			if (_mask.at<uchar>(p) == GC_PR_BGD || _mask.at<uchar>(p) == GC_PR_FGD) 
			{
				if ((_graph->what_segment(p.y*_mask.cols + p.x)==GraphCut::SOURCE))//如果属于SOURCE 就是可能前景
					_mask.at<uchar>(p) = GC_PR_FGD;
				else _mask.at<uchar>(p) = GC_PR_BGD;
			}
		}
	}
}

//Hist β=(2*<(Zm-Zn)^2)^0.5 八邻域邻近的计算~
double calcuBeta_Hist(const Mat& _img)
{
	double beta;
	double totalDiff = 0;
	for (int y = 0; y < _img.rows; y++) {
		for (int x = 0; x < _img.cols; x++) {
			Vec3f color = (Vec3f)_img.at<Vec3f>(y, x);
			if (x > 0) {
				Vec3f diff = color - (Vec3f)_img.at<Vec3f>(y, x - 1);
				totalDiff += diff.dot(diff);
			}
			if (y > 0 && x > 0) {
				Vec3f diff = color - (Vec3f)_img.at<Vec3f>(y - 1, x - 1);
				totalDiff += diff.dot(diff);
			}
			if (y > 0) {
				Vec3f diff = color - (Vec3f)_img.at<Vec3f>(y - 1, x);
				totalDiff += diff.dot(diff);
			}
			if (y > 0 && x < _img.cols - 1) {
				Vec3f diff = color - (Vec3f)_img.at<Vec3f>(y - 1, x + 1);
				totalDiff += diff.dot(diff);
			}
		}
	}
	totalDiff *= 2;
	if (totalDiff <= std::numeric_limits<double>::epsilon()) beta = 0;//避免太小呢
	else beta = 1.0 / (2 * totalDiff / (8 * _img.cols*_img.rows - 6 * _img.cols - 6 * _img.rows + 4));//计算期望
	return beta;
}

//Hist 平滑项V的预计算 其实这里不用用模板的。。。正常算就行
void calcuNWeight_Hist(const Mat& _img, Mat& _l, Mat& _ul, Mat& _u, Mat& _ur, double _beta, double _gamma) {
	const double gammaDiv = _gamma / std::sqrt(2.0f);// 原论文这里是没有考虑距离倒数了啊？难道这里有问题？
	_l.create(_img.size(), CV_64FC1);
	_ul.create(_img.size(), CV_64FC1);
	_u.create(_img.size(), CV_64FC1);
	_ur.create(_img.size(), CV_64FC1);
	for (int y = 0; y < _img.rows; y++) {
		for (int x = 0; x < _img.cols; x++) {
			Vec3f color = (Vec3f)_img.at<Vec3f>(y, x);
			if (x - 1 >= 0) {
				//Vec3f diff = vecDist<float,3>(color,(Vec3f)_img.at<Vec3f>(y, x - 1));
				Vec3f diff = vecSqrDist<float,3>(color,(Vec3f)_img.at<Vec3f>(y, x - 1));
				_l.at<double>(y, x) = _gamma * exp(-_beta * diff.dot(diff));
			}
			else _l.at<double>(y, x) = 0;
			if (x - 1 >= 0 && y - 1 >= 0) {
				//Vec3f diff = vecDist<float,3>(color,(Vec3f)_img.at<Vec3f>(y - 1, x - 1));
				Vec3f diff = vecSqrDist<float, 3>(color, (Vec3f)_img.at<Vec3f>(y - 1, x - 1));
				_ul.at<double>(y, x) = gammaDiv * exp(-_beta * diff.dot(diff));
				//_ul.at<double>(y, x) = _gamma * exp(-_beta * diff.dot(diff));

			}
			else _ul.at<double>(y, x) = 0;
			if (y - 1 >= 0) {
				//Vec3f diff = vecDist<float,3>(color,(Vec3f)_img.at<Vec3f>(y - 1, x));
				Vec3f diff = vecSqrDist<float, 3>(color, (Vec3f)_img.at<Vec3f>(y - 1, x));
				_u.at<double>(y, x) = _gamma * exp(-_beta * diff.dot(diff));
			}
			else _u.at<double>(y, x) = 0;
			if (x + 1 < _img.cols && y - 1 >= 0) {
				//Vec3f diff = vecDist<float,3>(color,(Vec3f)_img.at<Vec3f>(y - 1, x + 1));
				Vec3f diff = vecSqrDist<float, 3>(color, (Vec3f)_img.at<Vec3f>(y - 1, x + 1));
				_ur.at<double>(y, x) = gammaDiv * exp(-_beta * diff.dot(diff));
				//_ur.at<double>(y, x) = _gamma * exp(-_beta * diff.dot(diff));
			}
			else _ur.at<double>(y, x) = 0;
		}
	}
}

//Hist 将能量表达式转换成图从而利用max flow/min cut算法求解
GraphCut* getGraph_Hist(const Mat& _img, const Mat& _mask, const Hist& _bgdHist, const Hist& _fgdHist, double _lambda, const Mat& _l, const Mat& _ul, const Mat& _u, const Mat& _ur)
{
	int V_Count = _img.cols*_img.rows;//点集数量
	int E_Count = 2 * (4 * V_Count - 3 * _img.cols - 3 * _img.rows + 2);//边集数量
	GraphCut* _graph = new GraphCut(V_Count, E_Count);
	Point p;
	for (p.y = 0; p.y < _img.rows; p.y++)
	{
		for (p.x = 0; p.x < _img.cols; p.x++)
		{
			int vNum = _graph->add_node();
			Vec3f color = _img.at<Vec3f>(p);
			double wSource = 0, wSink = 0;
			if (_mask.at<uchar>(p) == GC_PR_BGD || _mask.at<uchar>(p) == GC_PR_FGD)
			{
				wSource = -log(_bgdHist.getweight(color));
				wSink = -log(_fgdHist.getweight(color));
			}
			else if (_mask.at<uchar>(p) == GC_BGD) wSink = _lambda;
			else wSource = _lambda;
			_graph->add_tweights(vNum, wSource, wSink);
			if (p.x > 0)
			{
				double w = _l.at<double>(p);
				_graph->add_edge(vNum, vNum - 1, w, w);
			}
			if (p.x > 0 && p.y > 0)
			{
				double w = _ul.at<double>(p);
				_graph->add_edge(vNum, vNum - _img.cols - 1, w, w);
			}
			if (p.y > 0)
			{
				double w = _u.at<double>(p);
				_graph->add_edge(vNum, vNum - _img.cols, w, w);
			}
			if (p.x < _img.cols - 1 && p.y > 0)
			{
				double w = _ur.at<double>(p);
				_graph->add_edge(vNum, vNum - _img.cols + 1, w, w);
			}
		}
	}
	return _graph;
}

//将能量表达式转换成图从而利用max flow/min cut算法求解
GraphCut* getGraph_Hist2(const Mat& _img, const Mat& _mask, const Hist& _bgdHist, const Hist& _fgdHist, double _lambda, const Mat& _l, const Mat& _ul, const Mat& _u, const Mat& _ur)
{
	int V_Count = _img.cols*_img.rows;//点集数量
	int E_Count = 2 * (4 * V_Count - 3 * _img.cols - 3 * _img.rows + 2);//边集数量
	GraphCut* _graph = new GraphCut(V_Count, E_Count);
	Point p;
	for (p.y = 0; p.y < _img.rows; p.y++)
	{
		for (p.x = 0; p.x < _img.cols; p.x++)
		{
			int vNum = _graph->add_node();
			Vec3b color = _img.at<Vec3b>(p);
			double wSource = 0, wSink = 0;
			if (_mask.at<uchar>(p) == GC_PR_BGD || _mask.at<uchar>(p) == GC_PR_FGD)
			{
				wSource = -log(_bgdHist.getweight(color));
				wSink = -log(_bgdHist.getweight(color));

			}
			else if (_mask.at<uchar>(p) == GC_BGD) wSink = _lambda;
			else wSource = _lambda;
			_graph->add_tweights(vNum, wSource, wSink);
			if (p.x > 0)
			{
				double w = _l.at<double>(p);
				_graph->add_edge(vNum, vNum - 1, w, w);
			}
			if (p.x > 0 && p.y > 0)
			{
				double w = _ul.at<double>(p);
				_graph->add_edge(vNum, vNum - _img.cols - 1, w, w);
			}
			if (p.y > 0)
			{
				double w = _u.at<double>(p);
				_graph->add_edge(vNum, vNum - _img.cols, w, w);
			}
			if (p.x < _img.cols - 1 && p.y > 0)
			{
				double w = _ur.at<double>(p);
				_graph->add_edge(vNum, vNum - _img.cols + 1, w, w);
			}
		}
	}
	return _graph;
}

//计算前背景各自像素数量，给颜色直方图使用
void cacluCount(const Mat& mask, int& fgdCount, int&bgdCount)
{
	Point p;
	for (p.y = 0; p.y < mask.rows; p.y++) {
		for (p.x = 0; p.x < mask.cols; p.x++) {
			uchar t = mask.at<uchar>(p);
			if (t == GC_BGD || t == GC_PR_BGD)bgdCount++;
			else fgdCount++;
		}
	}
}

double calcuGMM_energy(const Mat& _img, const Mat& _mask, const GMM& _bgdGMM, const GMM& _fgdGMM, double _lambda, const Mat& _l, const Mat& _ul, const Mat& _u, const Mat& _ur)
{
	double E = 0;
	Point p;
	for (p.y = 0; p.y < _img.rows; p.y++)
	{
		for (p.x = 0; p.x < _img.cols; p.x++)
		{
			Vec3b color = _img.at<Vec3b>(p);
			double wSource = 0, wSink = 0;
			if (_mask.at<uchar>(p) == GC_PR_BGD)
				E += -log(_bgdGMM.tWeight(color));
			else if (_mask.at<uchar>(p) == GC_PR_FGD)
				E += -log(_fgdGMM.tWeight(color));
			if (p.x > 0)
			{
				if (_mask.at<uchar>(p.y, p.x) == _mask.at<uchar>(p.y, p.x - 1))
					E += _l.at<double>(p);

			}
			if (p.x > 0 && p.y > 0)
			{
				if (_mask.at<uchar>(p.y, p.x) == _mask.at<uchar>(p.y - 1, p.x - 1))
					E += _ul.at<double>(p);
			}
			if (p.y > 0)
			{
				if (_mask.at<uchar>(p.y, p.x) == _mask.at<uchar>(p.y - 1, p.x))
					E += _u.at<double>(p);
			}
			if (p.x < _img.cols - 1 && p.y > 0)
			{
				if (_mask.at<uchar>(p.y, p.x) == _mask.at<uchar>(p.y, p.x + 1))
					E += _ur.at<double>(p);
			}
		}
	}
	return E;
}