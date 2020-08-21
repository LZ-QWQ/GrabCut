#include "utils.h"
#include "hist.h"

GrabCut2D::~GrabCut2D(void)
{
}

double GrabCut2D::GrabCut( cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect, cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel, int iterCount, int mode )
{
    //初始化
	Mat img = _img.getMat();
	Mat& mask = _mask.getMatRef();
	Mat& bgdModel = _bgdModel.getMatRef();
	Mat& fgdModel = _fgdModel.getMatRef();

	//模型等准备
	if(mode==GC_WITH_RECT)
		initMaskInRect(mask, rect, img.size());//因为交互操作产生的Mask是叠加了 可能前、背景的
	GMM bgdGMM(bgdModel);
	GMM fgdGMM(fgdModel);
	if (mode == GC_WITH_MASK || mode == GC_WITH_RECT)initGMM(img, mask, bgdGMM, fgdGMM);
	if (iterCount != 1)
	{
		//这里是为了计算energy
		cout << "iterCount 除1以外暂时都不行" << endl;
		return -1;//这个地方原则来讲不会遇到，真遇到了肯定有问题，先这样吧
	}
	
	//平滑项准备
	const double gamma = 50;
	const double beta = calcuBeta(img);
	Mat leftW, upleftW, upW, uprightW;
	calcuNWeight(img, leftW, upleftW, upW, uprightW, beta, gamma);

	Mat compIdxs(img.size(), CV_32SC1);//分配的GMM component

	//这个lambda是用来赋予确定的t-link、n-link权重的，
	//取值的话我认为是为了保证足够而设置的，9*gamma也保证了大于等于平滑项权重
	const double lambda = 9 * gamma;


	//迭代~
	for (int i = 0; i < iterCount; i++)
	{
		assignGMMS(img, mask, bgdGMM, fgdGMM, compIdxs);//第一步
		learnGMMs(img, mask, bgdGMM, fgdGMM, compIdxs);//第二步
		GraphCut *graph = getGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW);//把能量式变成图呀~
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
	//初始化
	
	Mat img = _img.getMat();
	Mat img3f=img.clone();
	img3f.convertTo(img3f, CV_32FC3, 1.0 / 255);
	Mat& mask = _mask.getMatRef();
	Mat& bgdModel = _bgdModel.getMatRef();//unused
	Mat& fgdModel = _fgdModel.getMatRef();//unused

	

	//模型等准备
	if (mode == GC_WITH_RECT)
		initMaskInRect(mask, rect, img.size());//因为交互操作产生的Mask是叠加了 可能前、背景的
	Hist bgdHist;
	Hist fgdHist;

	if (iterCount != 1)
	{
		cout << "iterCount 除1以外暂时都不行" << endl;
		return;
	}

	//平滑项准备
	Mat img3f_lab=img3f.clone();
	cvtColor(img3f, img3f_lab, COLOR_BGR2Lab);//img上转lab空间
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
	Mat compIdxs(img.size(), CV_32SC1);//分配的GMM component

	//这个lambda是用来赋予确定的t-link、n-link权重的，
	//取值的话我认为是为了保证足够而设置的，9*gamma也保证了大于等于平滑项权重
	const double lambda = 9 * gamma;

	

	//迭代~
	for (int i = 0; i < iterCount; i++)
	{
		//计算前背景各自像素数量，给颜色直方图使用
		int fgdCount = 0;
		int bgdCount = 0;
		cacluCount(mask, fgdCount, bgdCount);
		bgdHist.learnHist_BGD(img3f, mask, bgdCount);
		fgdHist.learnHist_FGD(img3f, mask, fgdCount);
		GraphCut* graph = nullptr;
		if (type == "RGB") graph = getGraph_Hist2(img, mask, bgdHist, fgdHist, lambda, leftW, upleftW, upW, uprightW);//把能量式变成图呀~
		else graph = getGraph_Hist(img3f_lab, mask, bgdHist, fgdHist, lambda, leftW, upleftW, upW, uprightW);//把能量式变成图呀~
		estimateSegmentation(graph, mask);//割！

		delete graph;
	}


}
