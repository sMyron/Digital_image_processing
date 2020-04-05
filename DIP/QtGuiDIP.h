#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_QtGuiDIP.h"
#include <qwidget.h>
#include <qgraphicsscene.h>
#include <qgraphicsview.h>  //graphicsview��
#include <qfiledialog.h>    //getopenfilename������
#include <qlabel.h>         //label��

#include <opencv40/opencv2/opencv.hpp>
using namespace cv;

class QtGuiDIP : public QMainWindow
{
	Q_OBJECT

public:
	QtGuiDIP(QWidget *parent = Q_NULLPTR);
	~QtGuiDIP();

private slots:
	void on_ReadFile_clicked();                 //��ͼ��
	void on_Flip_clicked();                     //��ת
	void on_ColorHist_clicked();                //��ɫֱ��ͼ
	void on_FFT_clicked();                      //FFT
	void on_HSI_clicked();                      //HSI
	void on_Gamma_clicked();                    //Gamma�任
	QImage cvGrayMat2QImage(const cv::Mat& mat);//CV�Ҷ�ͼ��QImage��ʾ
	void RGB2HSI(Mat &RGBImg, Mat &hsi);        //HSI��ɫ�ռ�
	void on_Gaussian_clicked();                 //��˹�˲�
	void on_Mean_clicked();                     //��ֵ�˲�
	void on_Sobel_clicked();                    //Sobel
	void on_Prewitt_clicked();                  //Prewitt
	Mat prewitt(Mat &imageP);                   //Prewitt�㷨
	void on_Laplacian_clicked();                //Laplacian
	Mat myLaplacian(Mat &imageL);               //Laplacian�㷨
	void on_Save_clicked();                     //����ͼ��
	void on_Scale_clicked();                    //����
	void on_ToGray_clicked();                   //RGBת�Ҷ�ͼ
	void on_OTSU_clicked();                     //OTSU��ֵ������
	void on_Dilate_clicked();                   //����
	void on_Erode_clicked();                    //��ʴ
	void on_OpenOperate_clicked();              //������ �ȸ�ʴ������
	void on_CloseOperate_clicked();             //������ �����ͺ�ʴ
private:
	Ui::QtGuiDIPClass ui;
	Mat image;               //ԭʼͼ��
	Mat imageLast;           //��һ����ͨ��ͼ��
	Mat imageLastC1;         //��һ����ͨ��ͼ��
	QLabel *label;
	QLabel *label_2;
	QLabel *labelLast;
	float gamma = 1;         //gamma�任������ʼ��  1Ϊ��ȱ任
	float scale = 1;         //���Ų�����ʼ��  1Ϊ��ԭͼ�ȴ�
};
