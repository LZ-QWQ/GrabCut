#include "BorderMatting.h"
BorderMatting::BorderMatting(){}
BorderMatting::~BorderMatting(){}
//�ж�x�Ƿ���l��rֱ�ӡ�
inline bool outrange(int _x, int _l, int _r){
	if (_x<_l || _x>_r)	return true;
	else return false;
}
//������ʼ����
void BorderMatting::init(const Mat& _img){
	rows = _img.rows;
	cols = _img.cols;
	sections = 0;
	areaCount = 0;
	contour.clear();
	strip.clear();
	vecds.clear();
}
//���� Canny �㷨���б�Ե��⣬�������� _rs �С�
void BorderDetection(const Mat& _img, Mat& _rs){
	Mat edges;
	Canny(_img, edges, 3, 9);
	edges.convertTo(_rs, CV_8UC1);
}
//����������������������������� contour ���й��졣
void BorderMatting::dfs(int _x, int _y, const Mat& _edge, Mat& _color){
	//��Ǳ������ĵ�
	_color.at<uchar>(_x, _y) = 255;
	para_point pt;
	pt.p.x = _x; pt.p.y = _y; //����
	pt.index = areaCount++;//��������ÿһ����������index
	pt.section = sections;//��������
	contour.push_back(pt); //��������vector
	//ö��(x,y)���ڵ�
	for (int i = 0; i < nstep; i++) {
		int zx = nx[i], zy = ny[i];
		int newx = _x + zx, newy = _y + zy;
		//����ͼ��Χ
		if (outrange(newx, 0, rows - 1) || outrange(newy, 0, cols - 1)) continue;
		//���������ϵĵ�
		if (_edge.at<uchar>(newx, newy) == 0)continue;
		//�Ѿ���������
		if (_color.at<uchar>(newx, newy) != 0)continue;
		//��(newx,newy)�������������ѱ�������
		dfs(newx, newy, _edge, _color);
	}
}
//������������������������в������㡣
void BorderMatting::ParameterizationContour(const Mat& _edge)
{
	int rows = _edge.rows, cols = _edge.cols;
	sections = 0; 
	areaCount = 0; 
	//�������
	Mat color(_edge.size(), CV_8UC1, Scalar(0));
	bool flag = false;
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			//(i,j)�������ϵĵ���δ��������
			if (_edge.at<uchar>(i, j) != 0 && color.at<uchar>(i, j) == 0){
				//������б�����ʹ��������һ
				dfs(i, j, _edge, color);
				sections++;
			}
}
//��ʼ��TU��������ͼ���洢��hash ֵ����������ֵ��
void BorderMatting::StripInit(const Mat& _mask){
	Mat color(_mask.size(), CV_32SC1, Scalar(0));//�������
	//���������������ѱ��TU�����TU�������򡪡�������Ӧ������������
	//��ʼ�����У��������������е�
	vector<point> queue;
	for (int i = 0; i < contour.size(); i++){
		inf_point ip;
		ip.p = contour[i].p; //����
		ip.dis = 0; //�������ĵ��ŷ�Ͼ���
		ip.area = contour[i].index; //��������
		strip[ip.p.x*COE + ip.p.y] = ip; //�������������key��hash��ֵΪ������
		queue.push_back(ip.p); //����������
		color.at<int>(ip.p.x, ip.p.y) = ip.area + 1; //������ǣ������+1
	}
	//���ѱ���TU����
	int l = 0;
	while (l < queue.size()){
		point p = queue[l++]; //ȡ����
		inf_point ip = strip[p.x*COE + p.y]; //��strip�еõ������Ϣ
		//ֻ����TU�ڵĵ�
		if (abs(ip.dis) >= stripwidth) break;
		int x = ip.p.x, y = ip.p.y;
		//ö�����ڵ�
		for (int i = 0; i < rstep; i++)	{
			int newx = x + rx[i], newy = y + ry[i];
			//����ͼ��Χ
			if (outrange(newx, 0, rows - 1) || outrange(newy, 0, cols - 1))	continue;
			inf_point nip;
			//����Ѿ���������
			if (color.at<int>(newx, newy) != 0)	continue;
			else nip.p.x = newx; nip.p.y = newy;
			nip.dis = abs(ip.dis) + 1;//ŷʽ����+1
			//����õ����ڱ�����ŷ�Ͼ���ȡ��
			if ((_mask.at<uchar>(newx, newy) & 1) != 1) nip.dis = -nip.dis;
			nip.area = ip.area;
			//����TU�С�
			strip[nip.p.x*COE + nip.p.y] = nip;
			queue.push_back(nip.p);
			//������ǣ������+1
			color.at<int>(newx, newy) = nip.area + 1;
		}
	}
}
//һά��˹�ܶȺ���
inline double Gaussian(double _x, double _delta, double _sigma){
	const double PI = 3.14159;
	double e = exp(-(pow(_x - _delta, 2.0) / (2.0*_sigma)));
	double rs = 1.0 / (pow(_sigma, 0.5)*pow(2.0*PI, 0.5))*e;
	return rs;
}
//�����й�ʽ15��1��
inline double ufunc(double _a, double _uf, double _ub){
	return (1.0 - _a)*_ub + _a*_uf;
}
//�����й�ʽ15��2��
inline double cfunc(double _a, double _cf, double _cb){
	return pow(1.0 - _a, 2.0)*_cb + pow(_a, 2.0)*_cf;
}
//sigmoid����,����soft step-function������ͼ6.c)
inline double Sigmoid(double _r, double _delta, double _sigma){
	double rs = -(_r - _delta) / _sigma;
	rs = exp(rs);
	rs = 1.0 / (1.0 + rs);
	return rs;
}
//����ĳһ�����������ֵ
inline double dataTermPoint(inf_point _ip, float _I, double _delta, double _sigma, double _uf, double _ub, double _cf, double _cb){
	double alpha = Sigmoid((double)_ip.dis / (double)stripwidth, _delta, _sigma);
	double D = Gaussian(_I, ufunc(alpha, _uf, _ub), cfunc(alpha, _cf, _cb));
	D = -log(D) / log(2.0);
	return D;
}
//�����һ����ʼ�㿪ʼ������������������֮��
double BorderMatting::dataTerm(int _index, point _p, double _uf, double _ub, double _cf, double _cb, double _delta, double _sigma, const Mat& _gray){
	vector<inf_point> queue;
	map<int, bool> color;
	double sum = 0;
	inf_point ip = strip[_p.x*COE + _p.y]; //��strip�л�ȡ���ĵ���Ϣ
	sum += dataTermPoint(ip, _gray.at<float>(ip.p.x, ip.p.y), _delta, _sigma, _uf, _ub, _cf, _cb);
	queue.push_back(ip);//�������
	color[ip.p.x*COE + ip.p.y] = true;//��Ǳ���
	//���ѱ�����pΪ���ĵ������
	int l = 0;
	while (l < queue.size())
	{
		inf_point ip = queue[l++];
		if (abs(ip.dis) >= stripwidth)break;
		int x = ip.p.x;
		int y = ip.p.y;
		//�������ڵ�
		for (int i = 0; i < rstep; i++)	{
			int newx = x + rx[i], newy = y + ry[i];
			if (outrange(newx, 0, rows - 1) || outrange(newy, 0, cols - 1))	continue;
			if (color[newx*COE + newy])	continue;
			inf_point newip = strip[newx*COE + newy];
			//������pΪ���ĵ������
			if (newip.area == _index) 
				sum += dataTermPoint(newip, _gray.at<float>(newx, newy), _delta, _sigma, _uf, _ub, _cf, _cb);
			queue.push_back(newip);//�������
			color[newx*COE + newy] = true;//��Ǳ���
		}
	}
	return sum;
}
/*����L*L�����ǰ������ֵ�ͷ���*/
void calMeanAndCov(point _p, const Mat& _gray, const Mat& _mask, double& _uf, double& _ub, double& _cf, double& _cb){
	int len = L;
	double sumf = 0, sumb = 0;
	int cntf = 0, cntb = 0;
	int rows = _gray.rows, cols = _gray.cols;
	//�����ֵ
	for (int x = _p.x - len; x <= _p.x + len; x++)
		for (int y = _p.y - len; y <= _p.y + len; y++)
			if (!(outrange(x, 0, rows - 1) || outrange(y, 0, cols - 1))){
				float g = _gray.at<float>(x, y);
				//����
				if ((_mask.at<uchar>(x, y) & 1) == 0){
					sumb += g;
					cntb++;
				}
				//ǰ��
				else {
					sumf += g;
					cntf++;
				}
			}

	_uf = (double)sumf / (double)cntf; //ǰ����ֵ
	_ub = (double)sumb / (double)cntb; //������ֵ
	//���㷽��
	_cf = 0;
	_cb = 0;
	for (int x = _p.x - len; x <= _p.x + len; x++)
		for (int y = _p.y - len; y <= _p.y + len; y++)
			if (!(outrange(x, 0, rows - 1) || outrange(y, 0, cols - 1))){
				float g = _gray.at<float>(x, y);
				//����
				if ((_mask.at<uchar>(x, y) & 1) == 0)
					_cb += pow(g - _ub, 2.0);
				//ǰ��
				else _cf += pow(g - _uf, 2.0);
			}
	_cf /= (double)cntf; //ǰ������
	_cb /= (double)cntb; //��������
}
//����sigma����ɢֵ
inline double sigma(int _level){ return 0.1*(_level); }
//����delta����ɢֵ
inline double delta(int level) { return 0.025*level; }
//����DP�㷨����������С������ sigma �� delta ��ֵ
void BorderMatting::EnergyMinimization(const Mat& _oriImg, const Mat& _mask){
	//ת��Ϊ�Ҷ�ͼ
	Mat gray;
	cvtColor(_oriImg, gray, COLOR_BGR2GRAY);
	gray.convertTo(gray, CV_32FC1, 1.0 / 255.0);
	//������С����ÿ�������delta��sigma
	//ö��������ÿһ���㣬���������������ĵ�
	for (int i = 0; i < contour.size(); i++) {
		para_point pp = contour[i];
		int index = pp.index;
		double uf, ub, cf, cb;
		//��L*L�����ǰ������ֵ�ͷ���
		calMeanAndCov(pp.p, gray, _mask, uf, ub, cf, cb);
		for (int d0 = 0; d0 < deltaLevels; d0++) //ö��delta
			for (int s0 = 0; s0 < sigmaLevels; s0++){ //ö��sigma
				double sigma0 = sigma(s0), delta0 = delta(d0);
				ef[index][d0][s0] = MAXNUM;
				//����term D
				double D = dataTerm(index, pp.p, uf, ub, cf, cb, delta0, sigma0, gray);
				//������������:termD + termV
				if (index == 0) {
					ef[index][d0][s0] = D;
					continue;
				}
				for (int d1 = 0; d1 < deltaLevels; d1++)//ö��index-1ʱ��delta
					for (int s1 = 0; s1 < sigmaLevels; s1++){//ö��index-1ʱ��sigma
						double delta1 = delta(d1), sigma1 = sigma(s1);
						double Vterm = 0;
						if (contour[i - 1].section == pp.section){//����һ������ͬһ����
							Vterm = varyTerm(delta0 - delta1, sigma0 - sigma1);
						}
						double rs = ef[index - 1][d1][s1] + Vterm + D;
						if (rs < ef[index][d0][s0]) {
							dands ds;
							ds.sigma = s1; ds.delta = d1;
							ef[index][d0][s0] = rs;
							rec[index][d0][s0] = ds;
						}
					}
			}
	}
	//����������Сֵ
	double minE = MAXNUM;
	dands ds;
	//��¼ÿ�������delta��sigma
	vecds = vector<dands>(areaCount);
	for (int d0 = 0; d0< deltaLevels; d0++)
		for (int s0 = 0; s0 < sigmaLevels; s0++)
		{
			if (ef[areaCount - 1][d0][s0] < minE) {
				minE = ef[areaCount - 1][d0][s0];
				ds.delta = d0;
				ds.sigma = s0;
			}
		}
	//��¼��������Сʱ��ÿ�������delta��sigma
	vecds[areaCount - 1] = ds;
	for (int i = areaCount - 2; i >= 0; i--){
		dands ds0 = vecds[i + 1];
		dands ds = rec[i + 1][ds0.delta][ds0.sigma];
		vecds[i] = ds;
	}
}
//����alpha��ֵ�����̫С����0���棬���̫����1����
inline double adjustA(double _a){
	if (_a < 0.01) return 0;
	if (_a > 0.99) return 1;
	return _a;
}
//����bfs��������alpha��ֵ
void BorderMatting::CalculateMask(Mat& _alphaMask, const Mat& _mask){
	_alphaMask = Mat(_mask.size(), CV_32FC1, Scalar(0));
	Mat visit(_mask.size(), CV_32SC1, Scalar(0));//�������
	//���������������ѱ���ͼ�񣬼���alpha
	//��ʼ�����У��������������е�											
	vector<inf_point> queue;
	for (int i = 0; i < contour.size(); i++){
		inf_point ip;
		ip.p = contour[i].p; //����
		ip.dis = 0; //�������ĵ��ŷ�Ͼ���
		ip.area = contour[i].index; //��������
		queue.push_back(ip); //����������
		visit.at<int>(ip.p.x, ip.p.y) = 1; //�������
		//����alpha
		dands ds = vecds[ip.area];
		double alpha = Sigmoid((double)ip.dis / (double)stripwidth, delta(ds.delta), sigma(ds.sigma));
		alpha = adjustA(alpha);//����alpha
		_alphaMask.at<float>(ip.p.x, ip.p.y) = (float)alpha;
	}
	//���ѱ�������
	int l = 0;
	while (l < queue.size()) {
		inf_point ip = queue[l++]; //ȡ����
		int x = ip.p.x, y = ip.p.y;
		for (int i = 0; i < rstep; i++){//ö�����ڵ�
			int newx = x + rx[i], newy = y + ry[i];
			if (outrange(newx, 0, rows - 1) || outrange(newy, 0, cols - 1))	continue;
			if (visit.at<int>(newx, newy) != 0)	continue;
			inf_point nip;
			nip.p.x = newx; nip.p.y = newy;
			nip.dis = abs(ip.dis) + 1;//ŷʽ����+1
			if ((_mask.at<uchar>(newx, newy) & 1) != 1)	nip.dis = -nip.dis;
			nip.area = ip.area;
			queue.push_back(nip); //�������
			visit.at<int>(newx, newy) = 1; //�������
			//����alpha
			dands ds = vecds[nip.area];
			double alpha = Sigmoid((double)nip.dis / (double)stripwidth, delta(ds.delta), sigma(ds.sigma));
			alpha = adjustA(alpha);//����alpha
			_alphaMask.at<float>(nip.p.x, nip.p.y) = (float)alpha;
		}
	}
}
//��ʾ������ͼƬ
void BorderMatting::display(const Mat& _originImage, const Mat& _alphaMask){
	vector<Mat> ch_img(3),ch_bg(3);
	//����ǰ���ͱ���ͨ�����豳��ɫΪ��ɫ
	Mat img;
	_originImage.convertTo(img, CV_32FC3, 1.0 / 255.0);
	cv::split(img, ch_img);
	Mat bg = Mat(img.size(), CV_32FC3, Scalar(0, 0, 0));
	cv::split(bg, ch_bg);
	//����alpha��ֵ�������͸�����Ժ��ͼ��
	ch_img[0] = ch_img[0].mul(_alphaMask) + ch_bg[0].mul(1.0 - _alphaMask);
	ch_img[1] = ch_img[1].mul(_alphaMask) + ch_bg[1].mul(1.0 - _alphaMask);
	ch_img[2] = ch_img[2].mul(_alphaMask) + ch_bg[2].mul(1.0 - _alphaMask);
	//�ϲ���ͨ��
	Mat res;
	cv::merge(ch_img, res);
	//��ʾ���
	cout << "boradmatting done~" << endl;
	namedWindow("img", WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);
	imshow("img", res);
	Mat res_4;
	vector<Mat>  src_channel, res_channel;
	//����ǰ���ͱ���ͨ�����豳��ɫΪ��ɫ
	//split(res, res_channel);
	//����alpha��ֵ�������͸�����Ժ��ͼ��
	split(_originImage, src_channel);//����splitһ�� ��Ϊ���������õ���0~255
	res_channel.push_back(src_channel[0]);
	res_channel.push_back(src_channel[1]);
	res_channel.push_back(src_channel[2]);
	Mat alphaMask(_originImage.size(), CV_32FC1);
	alphaMask.setTo(Scalar::all(255));
	Mat temp=alphaMask.mul(_alphaMask);
	temp.convertTo(temp, CV_8UC1);
	res_channel.push_back(temp);
	merge(res_channel, res_4);
	imwrite("./bordermatting.png", res_4);//�����ʱ���Ǳ���͸���ȵİ�~
}
//borderMatting�Ĺ��캯����Ҳ��������ṩ�Ľӿڡ�
void BorderMatting::borderMatting(const Mat& _originImage, const Mat& _mask, Mat& _alphaMask) {
	//��ʼ������
	init(_originImage);
	//�����������
	Mat edge = _mask & 1;
	edge.convertTo(edge, CV_8UC1, 255);
	BorderDetection(edge, edge);
	//����������
	ParameterizationContour(edge);
	//����TU����
	Mat tmask;
	_mask.convertTo(tmask, CV_8UC1);
	StripInit(tmask);
	//����DP�㷨����������С���õ�����ֵ
	EnergyMinimization(_originImage, _mask);
	//���ݵõ��Ĳ���ֵ����ÿ�����ص��alpha
	CalculateMask(_alphaMask, _mask);
	//��������ԣ����Խ�����΢�ĸ�˹�˲���ʹ�ý����ʾ��Χ����
	GaussianBlur(_alphaMask, _alphaMask, Size(7, 7), 9);
	//��ʾ borderMatting �Ľ��
	display(_originImage, _alphaMask);//����save�����棬2020.7.20 LZ
}