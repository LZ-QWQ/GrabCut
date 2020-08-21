#pragma once
#include"opencv.hpp"
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <math.h>
#define WINDOW_WIDTH 600
#define WINDOW_HEIGHT 600

using namespace cv;
using namespace std;

struct LineType
{
	char type;
	bool is_need_lined;	
	Scalar color;
};

class CPlot
{
public:	
	void DrawAxis (Mat *image); //��������
	void DrawData (Mat *image); //����
	int window_height; //���ڴ�С
	int window_width;


	vector< vector<Point2d> >dataset;	//�㼯��
	vector<LineType> lineTypeSet; //�ߵ�����
	
	//color
	Scalar backgroud_color;
	Scalar axis_color;
	Scalar text_color;

	Mat* Figure;

	// manual or automatic range
	bool custom_range_y;
	double y_max;
	double y_min;
	double y_scale;

	bool custom_range_x;
	double x_max;
	double x_min;
	double x_scale;
	
	//�߽��С
	int border_size;
		
	template<class T>
	void plot(T *y, size_t Cnt, Scalar color, char type = '*',bool is_need_lined = true);	
	template<class T>
	void plot(T *x, T *y, size_t Cnt, Scalar color, char type = '*',bool is_need_lined = true);
		
	void xlabel(string xlabel_name, Scalar label_color);
	void ylabel(string ylabel_name, Scalar label_color);
	//���ͼƬ�ϵ�����
	void clear();
	void title(string title_name,Scalar title_color); 
	
	CPlot();
	~CPlot();
		
};


////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
//���÷�����ƣ���˽�ʵ�ֲ��ֺ��������ַ���һ���ļ���
CPlot::CPlot()
{
	this->border_size = 40; //ͼ��Χ�߽�
	this->window_height = WINDOW_HEIGHT;
	this->window_width = WINDOW_WIDTH;
	this->Figure=new Mat(Size(this->window_height, this->window_width),CV_8UC3);
	memset(Figure->data, 255, sizeof(unsigned char)*Figure->cols*Figure->rows*3);
	//color
	this->backgroud_color = CV_RGB(255,255,255); //������ɫ
	this->axis_color = CV_RGB(0,0,0);//�����ɫ
	this->text_color = CV_RGB(255,0 ,0); //���ֺ�ɫ
	this->x_min = 0;
	this->x_max = 0;
	this->y_min = 0;
	this->y_max = 0;
}

CPlot::~CPlot()
{
	this->clear();
	Figure->release();
	delete Figure;
}

//�������
template<class T>
void CPlot::plot(T *X, T *Y, size_t Cnt, Scalar color, char type,bool is_need_lined)
{
	//�����ݽ��д洢
	T tempX, tempY;
	vector<Point2d>data;
	for(int i = 0; i < Cnt;i++)
	{
		tempX = X[i];
		tempY = Y[i];
		data.push_back( Point2d((double)tempX, (double)tempY) );
	}
	this->dataset.push_back(data);
	LineType LT;
	LT.type = type;
	LT.color = color;
	LT.is_need_lined = is_need_lined;
	this->lineTypeSet.push_back(LT);
	
	//printf("data count:%d\n", this->dataset.size());
	
	this->DrawData(this->Figure); //ÿ�ζ������»���
}

template<class T>
void CPlot::plot(T *Y, size_t Cnt, Scalar color, char type,bool is_need_lined)
{
	//�����ݽ��д洢
	T tempX, tempY;
	vector<Point2d>data;
	for(int i = 0; i < Cnt;i++)
	{
		tempX = i;
		tempY = Y[i];
		data.push_back(Point2d((double)tempX, (double)tempY));
	}
	this->dataset.push_back(data);
	LineType LT;
	LT.type = type;
	LT.color = color;
	LT.is_need_lined = is_need_lined;
	this->lineTypeSet.push_back(LT);
	this->DrawData(this->Figure);
}

void CPlot::clear()
{
	this->dataset.clear();
	this->lineTypeSet.clear();
}

void CPlot::title(string title_name,Scalar title_color = Scalar(0,0,0))
{
	int chw = 6, chh = 10; 
	Mat *image = this->Figure;
	int x = (this->window_width - 2 * this->border_size ) / 2 + this->border_size - ( title_name.size() / 2.0 ) * chw;
	int y = this->border_size / 2;
	putText( *image, title_name.c_str(), Point( x, y), FONT_HERSHEY_PLAIN, 1,title_color);
}

void CPlot::xlabel(string xlabel_name, Scalar label_color = Scalar(0,0,0))
{
	int chw = 6, chh = 10; 
	int bs = this->border_size;		
	int h = this->window_height;
	int w = this->window_width;
	// let x, y axies cross at zero if possible.
	double y_ref = this->y_min;
	if ( (this->y_max > 0 ) && ( this->y_min <= 0 ) )
	{
		y_ref = 0;
	}
	int x_axis_pos = h - bs - cvRound((y_ref - this->y_min) * this->y_scale);
	int x = this->window_width - this->border_size - chw * xlabel_name.size();
	int y = x_axis_pos + bs / 1.5;
	putText(*(this->Figure), xlabel_name.c_str(), Point( x, y), FONT_HERSHEY_PLAIN, 1,label_color);
}
void CPlot::ylabel(string ylabel_name, Scalar label_color = Scalar(0,0,0))
{

	int x = this->border_size;
	int y = this->border_size;
	putText(*(this->Figure), ylabel_name.c_str(), Point(x, y), FONT_HERSHEY_PLAIN, 1, label_color);
}

void CPlot::DrawAxis (Mat *image)
{

	Scalar axis_color = this->axis_color;
	
	int bs = this->border_size;		
	int h = this->window_height;
	int w = this->window_width;

	// size of graph
	int gh = h - bs * 2;
	int gw = w - bs * 2;

	// draw the horizontal and vertical axis
	// let x, y axies cross at zero if possible.
	double y_ref = this->y_min;
	if ( (this->y_max > 0 ) && ( this->y_min <= 0 ) )
	{
		y_ref = 0;
	}
	int x_axis_pos = h - bs - cvRound((y_ref - this->y_min) * this->y_scale);
	//X ��
	line(*image, Point(bs,     x_axis_pos), 
		           Point(w - bs, x_axis_pos),
				   axis_color);
	//Y ��
	line(*image, Point(bs, h - bs), 
		           Point(bs, h - bs - gh),
				   axis_color);

	// Write the scale of the y axis

	int chw = 6, chh = 10; 
	char text[16];

	// y max
	if ( (this->y_max - y_ref) > 0.05 * (this->y_max - this->y_min) )
	{
		_snprintf(text, sizeof(text)-1, "%.1f", this->y_max);
		putText(*image, text, Point(bs, bs / 2), FONT_HERSHEY_PLAIN, 1, this->text_color);
	}
	// y min
	if ( (y_ref - this->y_min) > 0.05 * (this->y_max - this->y_min) )
	{
		_snprintf(text, sizeof(text)-1, "%.1f", this->y_min);
		putText(*image, text, Point(bs, h - bs / 2), FONT_HERSHEY_PLAIN, 1,  this->text_color);
	}

	//��Y��Ŀ̶� ÿ�� scale_pixes ������
	//Y������
	double y_scale_pixes = chh * 2;
	for (int i = 0; i < ceil( (x_axis_pos - bs) / y_scale_pixes ) + 1; i++)
	{
		_snprintf(text, sizeof(text)-1, "%.1f", i * y_scale_pixes / this->y_scale );
		putText(*image, text, Point(bs / 5, x_axis_pos - i * y_scale_pixes), FONT_HERSHEY_PLAIN, 1, this->text_color);
	}
	//Y������
	for (int i = 1; i < ceil (( h - x_axis_pos - bs ) / y_scale_pixes ) + 1; i++)
	{
		_snprintf(text, sizeof(text)-1, "%.1f", -i * y_scale_pixes / this->y_scale );
		putText(*image, text, Point(bs / 5, x_axis_pos + i * y_scale_pixes), FONT_HERSHEY_PLAIN, 1, this->text_color);
	}

	// x_max
	_snprintf(text, sizeof(text)-1, "%.1f", this->x_max );
	putText(*image, text, Point(w - bs/2 - strlen(text) * chw, x_axis_pos), FONT_HERSHEY_PLAIN, 1, this->text_color);

	// x min
	_snprintf(text, sizeof(text)-1, "%.1f", this->x_min );
	putText(*image, text, Point(bs, x_axis_pos ), FONT_HERSHEY_PLAIN, 1, this->text_color);

	//��X��Ŀ̶� ÿ�� scale_pixes ������
	double x_scale_pixes = chw * 4;
	for (int i = 1; i < ceil( gw / x_scale_pixes ) + 1; i++)
	{
		_snprintf(text, sizeof(text)-1, "%.0f", this->x_min + i * x_scale_pixes / this->x_scale );
		putText(*image, text, Point(bs + i * x_scale_pixes - bs / 4, x_axis_pos + chh), FONT_HERSHEY_PLAIN, 1, this->text_color);
	}
}

//��Ӷ����͵�֧��
//TODO����δ��������
//���		����
//l          ֱ��	
//*          �� 
//.          �� 
//o          Ȧ 
//x          �� 
//+          ʮ�� 
//s          ���� 
void CPlot::DrawData (Mat *image)
{
	
	//this->x_min = this->x_max = this->dataset[0][0].x;
	//this->y_min = this->y_max = this->dataset[0][0].y;
	
	int bs = this->border_size;
	for(size_t i = 0; i < this->dataset.size(); i++)
	{
		for(size_t j = 0; j < this->dataset[i].size(); j++)
		{
			if(this->dataset[i][j].x < this->x_min)
			{
				this->x_min = this->dataset[i][j].x;
			}else if(this->dataset[i][j].x > this->x_max)
			{
				this->x_max = this->dataset[i][j].x;
			}
		
			if(this->dataset[i][j].y < this->y_min)
			{
				this->y_min = this->dataset[i][j].y;
			}else if(this->dataset[i][j].y > this->y_max)
			{
				this->y_max = this->dataset[i][j].y;
			}
		}
	}
	double x_range = this->x_max - this->x_min;
	double y_range = this->y_max - this->y_min;
	this->x_scale = (image->cols - bs*2)/ x_range;
	this->y_scale = (image->rows- bs*2)/ y_range;
	
	
	//����
	memset(image->data, 255, sizeof(unsigned char)*Figure->cols*Figure->rows*3);
	this->DrawAxis(image);
	
	//printf("x_range: %f y_range: %f\n", x_range, y_range);
	//���Ƶ�
	double tempX, tempY;
	Point prev_point, current_point;
	int radius = 3;
	int slope_radius = (int)( radius * 1.414 / 2 + 0.5);
	for(size_t i = 0; i < this->dataset.size(); i++)
	{
		for(size_t j = 0; j < this->dataset[i].size(); j++)
		{
			tempX = (int)((this->dataset[i][j].x - this->x_min)*this->x_scale);
			tempY = (int)((this->dataset[i][j].y - this->y_min)*this->y_scale);
			current_point = Point(bs + tempX, image->rows - (tempY + bs));
			
			if(this->lineTypeSet[i].type == 'l')
			{
				// draw a line between two points
				if (j >= 1)
				{
					line(*image, prev_point, current_point, lineTypeSet[i].color, 1, LINE_AA);
				}		
				prev_point = current_point;
			}else if(this->lineTypeSet[i].type == '.')
			{
				circle(*image, current_point, 1, lineTypeSet[i].color, -1, 8);
				if (lineTypeSet[i].is_need_lined == true)
				{
					if (j >= 1)
					{
						line(*image, prev_point, current_point, lineTypeSet[i].color, 1, LINE_AA);
					}		
					prev_point = current_point;
				}
			}else if(this->lineTypeSet[i].type == '*')
			{
				//��*
				line(*image, Point(current_point.x - slope_radius, current_point.y - slope_radius), 
			    Point(current_point.x + slope_radius, current_point.y + slope_radius), lineTypeSet[i].color, 1, 8);
					   
				line(*image, Point(current_point.x - slope_radius, current_point.y + slope_radius), 
			    Point(current_point.x + slope_radius, current_point.y - slope_radius), lineTypeSet[i].color, 1, 8);

				line(*image, Point(current_point.x - radius, current_point.y), 
				Point(current_point.x + radius, current_point.y), lineTypeSet[i].color, 1, 8);
					   
				line(*image, Point(current_point.x, current_point.y - radius), 
			    Point(current_point.x, current_point.y + radius), lineTypeSet[i].color, 1, 8);	 
				if (lineTypeSet[i].is_need_lined == true)
				{
					if (j >= 1)
					{
						line(*image, prev_point, current_point, lineTypeSet[i].color, 1, LINE_AA);
					}		
					prev_point = current_point;
				}
				
			}else if(this->lineTypeSet[i].type == 'o')
			{
				circle(*image, current_point, radius, this->text_color, 1, LINE_AA);
				if (lineTypeSet[i].is_need_lined == true)
				{
					if (j >= 1)
					{
						line(*image, prev_point, current_point, lineTypeSet[i].color, 1, LINE_AA);
					}		
					prev_point = current_point;
				}
			}else if(this->lineTypeSet[i].type == 'x')
			{
				line(*image, Point(current_point.x - slope_radius, current_point.y - slope_radius), 
			    Point(current_point.x + slope_radius, current_point.y + slope_radius), lineTypeSet[i].color, 1, 8);
					   
				line(*image, Point(current_point.x - slope_radius, current_point.y + slope_radius), 
			    Point(current_point.x + slope_radius, current_point.y - slope_radius), lineTypeSet[i].color, 1, 8);
				if (lineTypeSet[i].is_need_lined == true)
				{
					if (j >= 1)
					{
						line(*image, prev_point, current_point, lineTypeSet[i].color, 1, LINE_AA);
					}		
					prev_point = current_point;
				}
			}else if(this->lineTypeSet[i].type == '+')
			{
				line(*image, Point(current_point.x - radius, current_point.y), 
				Point(current_point.x + radius, current_point.y), lineTypeSet[i].color, 1, 8);
					   
				line(*image, Point(current_point.x, current_point.y - radius), 
			    Point(current_point.x, current_point.y + radius), lineTypeSet[i].color, 1, 8);
				if (lineTypeSet[i].is_need_lined == true)
				{
					if (j >= 1)
					{
						line(*image, prev_point, current_point, lineTypeSet[i].color, 1, LINE_AA);
					}		
					prev_point = current_point;
				}
			}else if(this->lineTypeSet[i].type == 's')
			{
				rectangle(*image, Point(current_point.x - slope_radius, current_point.y - slope_radius), 
			    Point(current_point.x + slope_radius, current_point.y + slope_radius), lineTypeSet[i].color, 1, 8);
				if (lineTypeSet[i].is_need_lined == true)
				{
					if (j >= 1)
					{
						line(*image, prev_point, current_point, lineTypeSet[i].color, 1, LINE_AA);
					}		
					prev_point = current_point;
				}
			}

		}
	}	
}


/**
�����ṩ�Ƿ���C����ʹ��ϰ�ߵ��÷��������ṩC++���ͣ����ٴ���Ĳ���
*/

class Plot : public CPlot
{
public:
	//�������������� ���μ�
	template<class T>
	void plot( vector<T> Y,Scalar color, char type = '*',bool is_need_lined = true);	
	template<class T>
	void plot(vector< Point_<T> > p,Scalar color, char type = '*',bool is_need_lined = true);
	//����һ��������C�汾�� IplImage ת����Mat
	Mat figure()
	{
		return Mat(*(this->Figure));
	}
};



template<class T>
void Plot::plot(vector<T> Y, Scalar color, char type,bool is_need_lined)
{
	//�����ݽ��д洢
	T tempX, tempY;
	vector<Point2d>data;
	for(int i = 0; i < Y.size();i++)
	{
		tempX = i;
		tempY = Y[i];
		data.push_back(Point2d((double)tempX, (double)tempY));
	}
	this->dataset.push_back(data);
	LineType LT;
	LT.type = type;
	LT.color = color;
	LT.is_need_lined = is_need_lined;
	this->lineTypeSet.push_back(LT);
	this->DrawData(this->Figure);
}

template<class T>
void Plot::plot(vector< Point_<T> > p, Scalar color, char type,bool is_need_lined)
{
	//�����ݽ��д洢
	T tempX, tempY;
	vector<Point2d>data;
	for(int i = 0; i < p.size();i++)
	{
		tempX = p[i].x;
		tempY = p[i].y;
		data.push_back( Point2d((double)tempX, (double)tempY) );
	}
	this->dataset.push_back(data);
	LineType LT;
	LT.type = type;
	LT.color = color;
	LT.is_need_lined = is_need_lined;
	this->lineTypeSet.push_back(LT);
	
	//printf("data count:%d\n", this->dataset.size());
	
	this->DrawData(this->Figure); //ÿ�ζ������»���
}
