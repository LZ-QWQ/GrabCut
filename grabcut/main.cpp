//http://www.cad.zju.edu.cn/home/gfzhang/course/computational-photography/proj1-grabcut/grabcut.html
//https://github.com/MatthewLQM/GrabCut
//https://github.com/jack-Dong/testPolt/  魔鬼画图
//参考

#include <iostream>
#include "GCApplication.h"
#include <ctime>

static void help()
{
	std::cout << "\nThis program demonstrates GrabCut segmentation -- select an object in a region\n"
		"and then grabcut will attempt to segment it out.\n"
		"Call:\n"
		"./grabcut <image_name>\n"
		"\nSelect a rectangular area around the object you want to segment\n" <<
		"\nHot keys: \n"
		"\tESC - quit the program\n"
		"\tr - restore the original image\n"
		"\tg - GMM\n"
		"\th - hist_RGB\n"
		"\tj - hist_Lab\n"
		"\ts - save grabcut image\n"
		"\tb - boradmatting(直接搬过来的，我也没研究过😓，很慢！！)\n"
		"\n"
		"\tleft mouse button - set rectangle\n"
		"\n"
		"\tCTRL+left mouse button - set GC_BGD pixels\n"
		"\tSHIFT+left mouse button - set CG_FGD pixels\n"
		"\n"
		"\tCTRL+right mouse button - set GC_PR_BGD pixels\n"
		"\tSHIFT+right mouse button - set CG_PR_FGD pixels\n"
		"用法：\n  grabcut.exe filename(同目录下加./)"<< endl;
}


GCApplication gcapp;

static void on_mouse( int event, int x, int y, int flags, void* param )
{
	gcapp.mouseClick( event, x, y, flags, param );
}


int main(int argc, char* argv[])
{
	string filename;
	if(argc==1)
		filename = "C:/Users/lz183/Desktop/呜呜呜/南开/媒体计算实验一轮考核/test1.jpg";
		//filename = "./test1.jpg";
	else if (argc == 2) filename = argv[1];
	else
	{
		cout << "参数至多为一个文件名（具体路径）" << endl;
		exit(EXIT_FAILURE);
	}
	clock_t startTime, endTime;
	double totalTime = 0;

	Mat image = imread( filename);
	if( image.empty() )
	{
		cout << "\n , couldn't read image filename " << filename << endl;
		return 1;
	}

	help();

	const string winName = "image";
	namedWindow( winName.c_str(), WINDOW_NORMAL|WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);
	//这个地方真的无语，按源码注释说应该是按比例的，可是没有？？为什么，出此下策
	resizeWindow(winName.c_str(), image.size());
	setMouseCallback( winName.c_str(), on_mouse, 0 );

	gcapp.setImageAndWinName( image, winName );
	gcapp.showImage();

	int iterCount = 0;
	int newIterCount = 0;
	double temp = 0;
	for(;;)
	{
		int c = waitKey(0);
		switch( (char) c )
		{
		case '\x1b'://Esc~
			cout << "Exiting ..." << endl;
			destroyWindow(winName.c_str());//关掉窗口说再见
			cout << "The total run time is: " << totalTime << "s" << endl;
			exit(EXIT_SUCCESS);
		case 'r':
			cout << endl;
			gcapp.reset();
			gcapp.showImage();
			cout << "The total run time is: " << totalTime << "s" << endl;
			totalTime = 0;
			break;
		case 's':
			gcapp.saveResultImage();
			cout << "save image done~" << endl;
			break;
		case 'b':
			cout << "<bordermatting... "<<endl;
			startTime = clock();
			gcapp.borderMatting();
			endTime = clock();
			temp = (double)(endTime - startTime) / CLOCKS_PER_SEC;
			cout << "...done~!  The bordermatting run time is: " << temp << "s" << endl;
			break;
		case 'g'://GMM
			startTime = clock();
			iterCount = gcapp.getIterCount();
			cout << "<" << iterCount << "... ";
			newIterCount = gcapp.nextIter_GMM();
			if( newIterCount > iterCount )
			{
				gcapp.showImage();
				cout << iterCount << ">" << endl;
			}

			endTime = clock();
			temp= (double)(endTime - startTime) / CLOCKS_PER_SEC;
			totalTime += temp;
			cout << "The run time is: " << temp << "s" << endl;
			break;
		case 'h'://histogram RGB gamma=50
			startTime = clock();
			iterCount = gcapp.getIterCount();
			cout << "<" << iterCount << "... ";
			newIterCount = gcapp.nextIter_Hist("RGB");
			if (newIterCount > iterCount)
			{
				gcapp.showImage();
				cout << iterCount << ">" << endl;
			}
			endTime = clock();
			temp = (double)(endTime - startTime) / CLOCKS_PER_SEC;
			totalTime += temp;
			cout << "The run time is: " << temp << "s" << endl;
			break;
		case 'j'://histogram Lab空间 gamma=1000	
			startTime = clock();
			iterCount = gcapp.getIterCount();
			cout << "<" << iterCount << "... ";
			newIterCount = gcapp.nextIter_Hist("Lab");
			if (newIterCount > iterCount)
			{
				gcapp.showImage();
				cout << iterCount << ">" << endl;
			}
			endTime = clock();
			temp = (double)(endTime - startTime) / CLOCKS_PER_SEC;
			totalTime += temp;
			cout << "The run time is: " << temp << "s" << endl;
			break;
		}
	}

	return 0;
}