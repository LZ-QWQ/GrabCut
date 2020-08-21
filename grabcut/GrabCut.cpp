#include "utils.h"
#include "hist.h"

GrabCut2D::~GrabCut2D(void)
{
}

double GrabCut2D::GrabCut( cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect, cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel, int iterCount, int mode )
{
    //��ʼ��
	Mat img = _img.getMat();
	Mat& mask = _mask.getMatRef();
	Mat& bgdModel = _bgdModel.getMatRef();
	Mat& fgdModel = _fgdModel.getMatRef();

	//ģ�͵�׼��
	if(mode==GC_WITH_RECT)
		initMaskInRect(mask, rect, img.size());//��Ϊ��������������Mask�ǵ����� ����ǰ��������
	GMM bgdGMM(bgdModel);
	GMM fgdGMM(fgdModel);
	if (mode == GC_WITH_MASK || mode == GC_WITH_RECT)initGMM(img, mask, bgdGMM, fgdGMM);
	if (iterCount != 1)
	{
		//������Ϊ�˼���energy
		cout << "iterCount ��1������ʱ������" << endl;
		return -1;//����ط�ԭ�����������������������˿϶������⣬��������
	}
	
	//ƽ����׼��
	const double gamma = 50;
	const double beta = calcuBeta(img);
	Mat leftW, upleftW, upW, uprightW;
	calcuNWeight(img, leftW, upleftW, upW, uprightW, beta, gamma);

	Mat compIdxs(img.size(), CV_32SC1);//�����GMM component

	//���lambda����������ȷ����t-link��n-linkȨ�صģ�
	//ȡֵ�Ļ�����Ϊ��Ϊ�˱�֤�㹻�����õģ�9*gammaҲ��֤�˴��ڵ���ƽ����Ȩ��
	const double lambda = 9 * gamma;


	//����~
	for (int i = 0; i < iterCount; i++)
	{
		assignGMMS(img, mask, bgdGMM, fgdGMM, compIdxs);//��һ��
		learnGMMs(img, mask, bgdGMM, fgdGMM, compIdxs);//�ڶ���
		GraphCut *graph = getGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW);//������ʽ���ͼѽ~
		estimateSegmentation(graph, mask);
		delete graph;
	}
	
	double temp_energy = calcuGMM_energy(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW);
	return temp_energy;
	
}

GrabCut2D_Hist::~GrabCut2D_Hist(void)
{
}

void GrabCut2D_Hist::GrabCut(cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect, cv::InputOutputArray _bgdModel, cv::InputOutputArray _fgdModel, int iterCount, int mode, const char* type)
{
	//��ʼ��
	
	Mat img = _img.getMat();
	Mat img3f=img.clone();
	img3f.convertTo(img3f, CV_32FC3, 1.0 / 255);
	Mat& mask = _mask.getMatRef();
	Mat& bgdModel = _bgdModel.getMatRef();//unused
	Mat& fgdModel = _fgdModel.getMatRef();//unused

	

	//ģ�͵�׼��
	if (mode == GC_WITH_RECT)
		initMaskInRect(mask, rect, img.size());//��Ϊ��������������Mask�ǵ����� ����ǰ��������
	Hist bgdHist;
	Hist fgdHist;

	if (iterCount != 1)
	{
		cout << "iterCount ��1������ʱ������" << endl;
		return;
	}

	//ƽ����׼��
	Mat img3f_lab=img3f.clone();
	cvtColor(img3f, img3f_lab, COLOR_BGR2Lab);//img��תlab�ռ�
	double gamma = 0;
	double beta = 0;	
	Mat leftW, upleftW, upW, uprightW;
	if (type == "RGB")
	{
		gamma = 50;
		beta = calcuBeta(img);
		calcuNWeight(img, leftW, upleftW, upW, uprightW, beta, gamma);		
	}
	else
	{
		gamma = 1000;
		beta = calcuBeta_Hist(img3f_lab);
		calcuNWeight_Hist(img3f_lab, leftW, upleftW, upW, uprightW, beta, gamma);		
	}
	Mat compIdxs(img.size(), CV_32SC1);//�����GMM component

	//���lambda����������ȷ����t-link��n-linkȨ�صģ�
	//ȡֵ�Ļ�����Ϊ��Ϊ�˱�֤�㹻�����õģ�9*gammaҲ��֤�˴��ڵ���ƽ����Ȩ��
	const double lambda = 9 * gamma;

	

	//����~
	for (int i = 0; i < iterCount; i++)
	{
		//����ǰ����������������������ɫֱ��ͼʹ��
		int fgdCount = 0;
		int bgdCount = 0;
		cacluCount(mask, fgdCount, bgdCount);
		bgdHist.learnHist_BGD(img3f, mask, bgdCount);
		fgdHist.learnHist_FGD(img3f, mask, fgdCount);
		GraphCut* graph = nullptr;
		if (type == "RGB") graph = getGraph_Hist2(img, mask, bgdHist, fgdHist, lambda, leftW, upleftW, upW, uprightW);//������ʽ���ͼѽ~
		else graph = getGraph_Hist(img3f_lab, mask, bgdHist, fgdHist, lambda, leftW, upleftW, upW, uprightW);//������ʽ���ͼѽ~
		estimateSegmentation(graph, mask);//�

		delete graph;
	}


}
