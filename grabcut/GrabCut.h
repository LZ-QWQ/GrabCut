#pragma once
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
enum
{
	GC_WITH_RECT  = 0, 
	GC_WITH_MASK  = 1, 
	GC_CUT        = 2  
};

class GrabCut2D
{
public:
	double GrabCut( cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect,
		cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel,
		int iterCount, int mode );  

	~GrabCut2D(void);
};

class GrabCut2D_Hist
{
public:
	void GrabCut(cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect,
		cv::InputOutputArray _bgdModel, cv::InputOutputArray _fgdModel,
		int iterCount, int mode, const char* type);

	~GrabCut2D_Hist(void);
};

