#include "GCApplication.h"
#include "plot.h"
/*
ԭ����������
#include <imgproc.hpp>

enum GrabCutClasses
{
	GC_BGD    = 0,  ����
	GC_FGD    = 1,  ǰ��
	GC_PR_BGD = 2,  ���ܵı��� �����Ŀ���
	GC_PR_FGD = 3,  ���ܵ�ǰ�� �����Ŀ���
};
*/
const string GCApplication::GMM_loss_Name = "GMM_loss";

GCApplication::GCApplication()
{	
	namedWindow(GMM_loss_Name.c_str(), WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);
}


//Set value for the class
void GCApplication::reset()
{
	if( !mask.empty() )
		mask.setTo(Scalar::all(GC_BGD));
	bgdPxls.clear(); fgdPxls.clear();
	prBgdPxls.clear();  prFgdPxls.clear();

	GMM_energys.clear();

	isInitialized_GMM = false;
	isInitialized_Hist = false;
	rectState = NOT_SET;    
	lblsState = NOT_SET;
	prLblsState = NOT_SET;
	iterCount = 0;
}

//Set image and window name
void GCApplication::setImageAndWinName( const Mat& _image, const string& _winName  )
{
	if( _image.empty() || _winName.empty() )
		return;
	image = &_image;
	winName = &_winName;
	mask.create( image->size(), CV_8UC1);
	reset();
}

//Show the result image
void GCApplication::showImage() const
{
	if( image->empty() || winName->empty() )
		return;

	Mat res;
	Mat binMask;
	if (!isInitialized_GMM && !isInitialized_Hist)
		image->copyTo(res);
	else
	{
		//����ط��Ҳ�֪��ԭ����ʲô��imshow������ʾ��͸����ͨ����ͼƬ�������Ҿ�ֻ��������
		getBinMask( mask, binMask );
		//image->copyTo( res, binMask );  //show the GrabCuted image
		vector<Mat> ch_img(3), ch_bg(3);
		//����ǰ���ͱ���ͨ�����豳��ɫΪ��ɫ
		split(*image, ch_img);
		Mat bg = Mat((*image).size(), CV_8UC3, Scalar(255, 255, 255));
		split(bg, ch_bg);
		//����alpha��ֵ�������͸�����Ժ��ͼ��
		ch_img[0] = ch_img[0].mul(binMask) + ch_bg[0].mul(1.0 - binMask);
		ch_img[1] = ch_img[1].mul(binMask) + ch_bg[1].mul(1.0 - binMask);
		ch_img[2] = ch_img[2].mul(binMask) + ch_bg[2].mul(1.0 - binMask);
		merge(ch_img, res);

	}

	vector<Point>::const_iterator it;
	//Using four different colors show the point which have been selected
	//�Ҿ���Ϊ���Ͻ�Ӧ������prob��ȷ�����ȱ�����ǰ��
	for (it = prBgdPxls.begin(); it != prBgdPxls.end(); ++it)
		circle(res, *it, radius, LIGHTBLUE, thickness);
	for (it = prFgdPxls.begin(); it != prFgdPxls.end(); ++it)
		circle(res, *it, radius, PINK, thickness);

	for( it = bgdPxls.begin(); it != bgdPxls.end(); ++it )  
		circle( res, *it, radius, BLUE, thickness );
	for( it = fgdPxls.begin(); it != fgdPxls.end(); ++it )  
		circle( res, *it, radius, GREEN, thickness );
	

	//Draw the rectangle
	if( rectState == IN_PROCESS || rectState == SET )
		rectangle( res, Point( rect.x, rect.y ), Point(rect.x + rect.width, rect.y + rect.height ), RED, 2);

	imshow( *winName, res );
	if (isInitialized_GMM)//Ҳ����GMM�����е�ʱ�򣡣�
	{
		CPlot plot;
		plot.x_max = GMM_energys.size();
		plot.x_min = 0;
		plot.y_max = *max_element(GMM_energys.begin(), GMM_energys.end());
		plot.y_min = *min_element(GMM_energys.begin(), GMM_energys.end());
		plot.axis_color = Scalar(0, 255, 0);
		plot.text_color = Scalar(255, 0, 255);
		int n = GMM_energys.size();

		double* arr=new double[n];
		double *x = new double[n];
		for (int i = 0; i < n; i++)
		{
			arr[i] = GMM_energys[i];
			x[i] = i;
		}
		plot.plot(x,arr, GMM_energys.size(), CV_RGB(0, 0, 0),'l');
		plot.title("GMM_loss"); //�����趨���� ֻ����Ӣ�� ���Ļ����� �н�����������Ǻ��鷳
		plot.xlabel("iter", Scalar(255, 255, 0));
		plot.ylabel("energy", Scalar(255, 255, 0));
		imshow(GMM_loss_Name, *plot.Figure);

		//��������������ʵ�ĵ������ı������ã���������Ū��
		imwrite(save__GMMenergy_filename, *plot.Figure);
	}
}

void GCApplication::saveResultImage() const
{
	if (image->empty() || winName->empty())
		return;

	Mat res;
	Mat binMask;
	if (!isInitialized_GMM&&!isInitialized_Hist)
		image->copyTo(res);
	else
	{
		getBinMask(mask, binMask);
		//image->copyTo(res, binMask);  //show the GrabCuted image

		vector<Mat> src_img, res_channel;
		//����ǰ���ͱ���ͨ�����豳��ɫΪ��ɫ
		split(*image, src_img);
		//split(res, res_channel);
		//����alpha��ֵ�������͸�����Ժ��ͼ��
		res_channel.push_back(src_img[0]);
		res_channel.push_back(src_img[1]);
		res_channel.push_back(src_img[2]);
		Mat alphaMask((*image).size(), CV_8UC1);
		alphaMask.setTo(Scalar::all(255));
		res_channel.push_back(alphaMask.mul(binMask));

		//�ϲ���ͨ��  �Ա���͸���ķ�ʽ����
		merge(res_channel, res);
	}

	imwrite(save_filename, res);
}


//Using rect initialize the pixel 
void GCApplication::setRectInMask()
{
	assert( !mask.empty() );
	mask.setTo( GC_BGD );   //GC_BGD == 0
	rect.x = max(0, rect.x);
	rect.y = max(0, rect.y);
	rect.width = min(rect.width, image->cols-rect.x);
	rect.height = min(rect.height, image->rows-rect.y);
	(mask(rect)).setTo( Scalar(GC_PR_FGD) );    //GC_PR_FGD == 3 
}

//Lbls ��ɶ��˼��
void GCApplication::setLblsInMask( int flags, Point p, bool isPr )
{
	vector<Point> *bpxls, *fpxls;
	uchar bvalue, fvalue;
	if( !isPr ) //Points which are sure being FGD or BGD
	{
		bpxls = &bgdPxls;
		fpxls = &fgdPxls;
		bvalue = GC_BGD;    //0
		fvalue = GC_FGD;    //1
	}
	else    //Probably FGD or Probably BGD
	{
		bpxls = &prBgdPxls;
		fpxls = &prFgdPxls;
		bvalue = GC_PR_BGD; //2
		fvalue = GC_PR_FGD; //3
	}
	if( flags & BGD_KEY )
	{
		bpxls->push_back(p);
		circle( mask, p, radius, bvalue, thickness );   //Set point value = 2
	}
	if( flags & FGD_KEY )
	{
		fpxls->push_back(p);
		circle( mask, p, radius, fvalue, thickness );   //Set point value = 3
	}
}


//Mouse Click Function: flags work with CV_EVENT_FLAG 
void GCApplication::mouseClick( int event, int x, int y, int flags, void* )
{
	switch( event )
	{
	case EVENT_LBUTTONDOWN: // Set rect or GC_BGD(GC_FGD) labels
		{
			bool isb = (flags & BGD_KEY) != 0,//��Ϊ��Щ��������2�Ĵη���ֵ�ģ����Կ�����λ�������ж�
				isf = (flags & FGD_KEY) != 0;
			if( rectState == NOT_SET && !isb && !isf )//Only LEFT_KEY pressed
			{
				rectState = IN_PROCESS; //Be drawing the rectangle
				rect = Rect( x, y, 0, 0 );
			}
			//������жϷ�ʽ��֤����;��������brush�أ�
			if ( (isb || isf) && rectState == SET ) //Set the BGD/FGD(labels).after press the "ALT" key or "SHIFT" key,and have finish drawing the rectangle
				lblsState = IN_PROCESS;
		}
		break;
	case EVENT_RBUTTONDOWN: // Set GC_PR_BGD(GC_PR_FGD) labels
		{
			bool isb = (flags & BGD_KEY) != 0,
				isf = (flags & FGD_KEY) != 0;
			if ( (isb || isf) && rectState == SET ) //Set the probably FGD/BGD labels
				prLblsState = IN_PROCESS;
		}
		break;
	case EVENT_LBUTTONUP:
		if( rectState == IN_PROCESS )
		{
			rect = Rect( Point(rect.x, rect.y), Point(x,y) );   //After draw the rectangle
			rectState = SET;
			setRectInMask();
			assert( bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty() );
			showImage();
		}
		if( lblsState == IN_PROCESS )   
		{
			setLblsInMask(flags, Point(x,y), false);    // Draw the FGD points
			lblsState = SET;
			showImage();
		}
		break;
	case EVENT_RBUTTONUP:
		if( prLblsState == IN_PROCESS )
		{
			setLblsInMask(flags, Point(x,y), true); //Draw the BGD points
			prLblsState = SET;
			showImage();
		}
		break;
	case EVENT_MOUSEMOVE:
		if( rectState == IN_PROCESS )
		{
			rect = Rect( Point(rect.x, rect.y), Point(x,y) );
			assert( bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty() );
			showImage();   //Continue showing image
		}
		else if( lblsState == IN_PROCESS )
		{
			setLblsInMask(flags, Point(x,y), false);
			showImage();
		}
		else if( prLblsState == IN_PROCESS )
		{
			setLblsInMask(flags, Point(x,y), true);
			showImage();
		}
		break;
	}
}

//Execute GrabCut algorithm��and return the iter count.
int GCApplication::nextIter_GMM()
{
	double temp_energy = 0;
	if (isInitialized_Hist)
	{
		cout << "��������Histogramģ�飬reset���ٳ���" << endl;
		return iterCount;
	}
	if( isInitialized_GMM)
		temp_energy=gc.GrabCut(*image, mask, rect, bgdModel, fgdModel,1,GC_CUT);
	else
	{
		if (rectState != SET)
		{
			cout << "rect must be determined>" << endl;
			return iterCount;
		}

		if( lblsState == SET || prLblsState == SET )
			temp_energy=gc.GrabCut( *image, mask, rect, bgdModel, fgdModel, 1, GC_WITH_MASK );
		else
			temp_energy=gc.GrabCut(*image, mask, rect, bgdModel, fgdModel,1,GC_WITH_RECT);
		isInitialized_GMM = true;
	}
	

	bgdPxls.clear(); fgdPxls.clear();
	prBgdPxls.clear(); prFgdPxls.clear();

	cout << "��" << iterCount << "�ε�������Ϊ��" << temp_energy << endl;
	GMM_energys.push_back(temp_energy); 
	iterCount++;
	return iterCount;
}

int GCApplication::nextIter_Hist(const char* type)
{
	//����type��Hist������ͻ����
	if (type != "RGB" && type != "Lab")
	{
		cout << "��ɫֱ��ͼtypeΪRGB��Lab" << endl;
		return iterCount;
	}
	if (isInitialized_GMM)
	{
		cout << "��������Histogramģ�飬reset���ٳ���" << endl;
		return iterCount;
	}
	if (isInitialized_Hist)
		gch.GrabCut(*image, mask, rect, bgdModel, fgdModel, 1, GC_CUT,type);
	else
	{
		if (rectState != SET)
		{
			cout << "rect must be determined>" << endl;
			return iterCount;
		}

		if (lblsState == SET || prLblsState == SET)
			gch.GrabCut(*image, mask, rect, bgdModel, fgdModel, 1, GC_WITH_MASK,type);
		else
			gch.GrabCut(*image, mask, rect, bgdModel, fgdModel, 1, GC_WITH_RECT,type);
		isInitialized_Hist = true;
	}
	iterCount++;

	bgdPxls.clear(); fgdPxls.clear();
	prBgdPxls.clear(); prFgdPxls.clear();

	return iterCount;
}

void GCApplication::borderMatting() {
	bm.borderMatting(*image, mask, alphaMask);
}