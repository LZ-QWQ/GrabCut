#pragma once
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "GrabCut.h"
#include "BorderMatting.h"
#include <iostream>

using namespace std;
using namespace cv;

const Scalar BLUE = Scalar(255,0,0); // Background 
const Scalar GREEN = Scalar(0,255,0);//Foreground
const Scalar LIGHTBLUE = Scalar(255,255,160);//ProbBackground
const Scalar PINK = Scalar(230,130,255); //ProbBackground
const Scalar RED = Scalar(0,0,255);//color of Rectangle

const int BGD_KEY = EVENT_FLAG_CTRLKEY;// When press "CTRL" key,the value of flags return.
const int FGD_KEY = EVENT_FLAG_SHIFTKEY;// When press "SHIFT" key, the value of flags return.


//Copy the value of comMask to binMask
static void getBinMask( const Mat& comMask, Mat& binMask )
{
	if( comMask.empty() || comMask.type()!=CV_8UC1 )
		CV_Error( Error::StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)" );
	if( binMask.empty() || binMask.rows!=comMask.rows || binMask.cols!=comMask.cols )
		binMask.create( comMask.size(), CV_8UC1 );
	binMask = comMask & 1;
}


class GCApplication
{
public:
	GCApplication();//构造函数里搞一个窗口用于输出可能的GMM energy （其实就算是loss吧）
	enum{ NOT_SET = 0, IN_PROCESS = 1, SET = 2 };//用于控制交互的，，(((φ(◎ロ◎;)φ)))
	static const int radius = 2;//brush的半径~
	static const int thickness = -1;//-1是填充
	static const string GMM_loss_Name;//GMM_loss 的窗口

	void reset();
	void setImageAndWinName( const Mat& _image, const string& _winName );
	void showImage() const;
	void mouseClick( int event, int x, int y, int flags, void* param );
	int nextIter_GMM();
	int nextIter_Hist(const char* type);
	int getIterCount() const { return iterCount; }
	void saveResultImage() const;
	void borderMatting();
	vector<double> GMM_energys;

private:
	void setRectInMask();
	void setLblsInMask( int flags, Point p, bool isPr );

	const string save_filename="./grabcut.png";
	const string save__GMMenergy_filename = "./GMM_energy.png";
	const string* winName;
	const Mat* image;
	Mat mask,alphaMask;
	Mat bgdModel, fgdModel;

	uchar rectState, lblsState, prLblsState;
	bool isInitialized_GMM;
	bool isInitialized_Hist;

	Rect rect;
	vector<Point> fgdPxls, bgdPxls, prFgdPxls, prBgdPxls;
	int iterCount;
	GrabCut2D gc;
	GrabCut2D_Hist gch;
	BorderMatting bm;
};


