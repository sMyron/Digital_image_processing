#pragma once
#include <iostream>
#include <opencv40/opencv2/opencv.hpp>
using namespace cv;
using namespace std;

class CalcHistogram {
private:
	int histSize[3];//直方图数量
	float hranges[2];//h通道最大最小值
	float sranges[2];
	float vranges[2];
	const float *ranges[3];//各通道范围
	int channels[3];
	int dims;

public:
	CalcHistogram(int hbins = 30, int sbins = 32, int vbins = 32)
	{
		histSize[0] = hbins;
		histSize[1] = sbins;
		histSize[2] = vbins;
		hranges[0] = 0; hranges[1] = 180;
		sranges[0] = 0; sranges[1] = 256;
		vranges[0] = 0; vranges[1] = 256;
		ranges[0] = hranges;
		ranges[1] = sranges;
		ranges[2] = vranges;
		channels[0] = 0;
		channels[1] = 1;
		channels[2] = 2;
		dims = 3;
	}
	
	Mat getHistogram(const Mat &image);
	void getHistogramImage(const Mat &image);

};