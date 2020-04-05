#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_QtGuiDIP.h"
#include <qwidget.h>
#include <qgraphicsscene.h>
#include <qgraphicsview.h>  //graphicsview类
#include <qfiledialog.h>    //getopenfilename类申明
#include <qlabel.h>         //label类

#include <opencv40/opencv2/opencv.hpp>
using namespace cv;

class QtGuiDIP : public QMainWindow
{
	Q_OBJECT

public:
	QtGuiDIP(QWidget *parent = Q_NULLPTR);
	~QtGuiDIP();

private slots:
	void on_ReadFile_clicked();                 //打开图像
	void on_Flip_clicked();                     //翻转
	void on_ColorHist_clicked();                //颜色直方图
	void on_FFT_clicked();                      //FFT
	void on_HSI_clicked();                      //HSI
	void on_Gamma_clicked();                    //Gamma变换
	QImage cvGrayMat2QImage(const cv::Mat& mat);//CV灰度图的QImage表示
	void RGB2HSI(Mat &RGBImg, Mat &hsi);        //HSI颜色空间
	void on_Gaussian_clicked();                 //高斯滤波
	void on_Mean_clicked();                     //均值滤波
	void on_Sobel_clicked();                    //Sobel
	void on_Prewitt_clicked();                  //Prewitt
	Mat prewitt(Mat &imageP);                   //Prewitt算法
	void on_Laplacian_clicked();                //Laplacian
	Mat myLaplacian(Mat &imageL);               //Laplacian算法
	void on_Save_clicked();                     //保存图像
	void on_Scale_clicked();                    //缩放
	void on_ToGray_clicked();                   //RGB转灰度图
	void on_OTSU_clicked();                     //OTSU二值化处理
	void on_Dilate_clicked();                   //膨胀
	void on_Erode_clicked();                    //腐蚀
	void on_OpenOperate_clicked();              //开运算 先腐蚀后膨胀
	void on_CloseOperate_clicked();             //闭运算 先膨胀后腐蚀
private:
	Ui::QtGuiDIPClass ui;
	Mat image;               //原始图像
	Mat imageLast;           //上一步三通道图像
	Mat imageLastC1;         //上一步单通道图像
	QLabel *label;
	QLabel *label_2;
	QLabel *labelLast;
	float gamma = 1;         //gamma变换参数初始化  1为恒等变换
	float scale = 1;         //缩放参数初始化  1为与原图等大
};
