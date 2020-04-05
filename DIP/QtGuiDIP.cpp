#include "QtGuiDIP.h"
#include "ColorHistogram.h"
#include <math.h>
#include <iostream>
#include <fstream>
using namespace cv;
using namespace std;

//构造函数
QtGuiDIP::QtGuiDIP(QWidget *parent)
	: QMainWindow(parent)
{
	
	QImage _image;
	_image.load("./cute.jpg");//背景图
	setAutoFillBackground(true);//自动填充
	QPalette pal(palette());
	pal.setBrush(QPalette::Window, QBrush(_image.scaled(size(), 
		Qt::IgnoreAspectRatio, Qt::SmoothTransformation)));
	setPalette(pal);//设置

	ui.setupUi(this);
}

//析构函数
QtGuiDIP::~QtGuiDIP() {

}

//打开图像
void QtGuiDIP::on_ReadFile_clicked() {
	QString fileName;
	fileName = QFileDialog::getOpenFileName(this,
		tr("Choose the Image"),
		"",
		tr("Images(*.png *.bmp *.jpg *.tif *.GIF)"));
	if (fileName.isEmpty()) {
		return;
	}
	else {
		string str = fileName.toStdString();
		image = imread(str);
		//赋值过后imageLast就成了RGB格式，而不是又边image的BGR！！！！特别注意!!
		imageLast = image;
		//读取进来的图片是BGR，得转成RGB，QImage才能够正常表示
		//imshow的则是BGR格式才能够正常表示
		cvtColor(image, image, COLOR_BGR2RGB);
		cv::resize(image, image, Size(220, 250));

		//imageLast已经是RGB格式，不需要下边这行代码
		//cvtColor(imageLast, imageLast, COLOR_BGR2RGB);
		cv::resize(imageLast, imageLast, Size(220, 250));

		QImage img1 = QImage((const unsigned char*)(image.data), image.cols, image.rows, QImage::Format_RGB888);
		label = new QLabel();
		label->setPixmap(QPixmap::fromImage(img1));
		label->resize(QSize(img1.width(), img1.height()));
		ui.scrollArea->setWidget(label);
		//imshow("yuantu", image);
		//imshow("shangyibu", imageLast);
		//waitKey(0);
		
	}
}

//翻转
void QtGuiDIP::on_Flip_clicked() {
	QImage imglast = QImage((const unsigned char*)(imageLast.data), imageLast.cols, imageLast.rows, QImage::Format_RGB888);
	labelLast = new QLabel();
	labelLast->setPixmap(QPixmap::fromImage(imglast));
	labelLast->resize(QSize(imglast.width(), imglast.height()));
	ui.scrollArea_3->setWidget(labelLast);
	
	Mat image_Flip = imageLast;
	flip(image_Flip, image_Flip, 4);
	imageLast = image_Flip;
	
	QImage img = QImage((const unsigned char*)(image_Flip.data), image_Flip.cols, image_Flip.rows, QImage::Format_RGB888);
	label_2 = new QLabel();
	label_2->setPixmap(QPixmap::fromImage(img));
	label_2->resize(QSize(img.width(), img.height()));
	ui.scrollArea_2->setWidget(label_2);
}

//颜色直方图
void QtGuiDIP::on_ColorHist_clicked() {
	Mat dst;
	Mat image_Hist;
	//image_Hist是RGB
	image_Hist = image;
	/// 分割成3个单通道图像 ( R, G 和 B )
	vector<Mat> rgb_planes;
	split(image_Hist, rgb_planes);

	/// 设定bin数目
	int histSize = 255;

	/// 设定取值范围 ( R,G,B) )
	float range[] = { 0, 255 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat r_hist, g_hist, b_hist;

	/// 计算直方图:
	calcHist(&rgb_planes[0], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&rgb_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&rgb_planes[2], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);

	// 创建直方图画布
	int hist_w = 400; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_w, hist_h, CV_8UC3, Scalar(0, 0, 0));

	/// 将直方图归一化到范围 [ 0, histImage.rows ]
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// 在直方图画布上画出直方图
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}
	cv::resize(histImage, histImage, Size(300, 200));

	QImage img1 = QImage((const unsigned char*)(histImage.data), histImage.cols, histImage.rows, QImage::Format_RGB888);
	label = new QLabel();
	label->setPixmap(QPixmap::fromImage(img1));
	label->resize(QSize(img1.width(), img1.height()));
	ui.scrollArea_4->setWidget(label);
}

//FFT
void QtGuiDIP::on_FFT_clicked() {
	Mat image_FFT;
	image_FFT = image;

	cvtColor(image_FFT, image_FFT, COLOR_RGB2GRAY);
	int m = getOptimalDFTSize(image_FFT.rows);
	int n = getOptimalDFTSize(image_FFT.cols);
	Mat padded;                 //以0填充输入图像矩阵
	//填充输入图像I，输入矩阵为padded，上方和左方不做填充处理
	copyMakeBorder(image_FFT, padded, 0, m - image_FFT.rows, 0, n - image_FFT.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(),CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);     //将planes融合合并成一个多通道数组complexI

	dft(complexI, complexI);        //进行傅里叶变换

	//计算幅值，转换到对数尺度(logarithmic scale)
	//=> log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	split(complexI, planes);        //planes[0] = Re(DFT(I),planes[1] = Im(DFT(I))
									//即planes[0]为实部,planes[1]为虚部
	magnitude(planes[0], planes[1], planes[0]);     //planes[0] = magnitude
	Mat magI = planes[0];

	magI += Scalar::all(1);
	log(magI, magI);                //转换到对数尺度(logarithmic scale)

	//如果有奇数行或列，则对频谱进行裁剪
	magI = magI(Rect(0, 0, magI.cols&-2, magI.rows&-2));

	//重新排列傅里叶图像中的象限，使得原点位于图像中心
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	Mat q0(magI, Rect(0, 0, cx, cy));       //左上角图像划定ROI区域
	Mat q1(magI, Rect(cx, 0, cx, cy));      //右上角图像
	Mat q2(magI, Rect(0, cy, cx, cy));      //左下角图像
	Mat q3(magI, Rect(cx, cy, cx, cy));     //右下角图像

	//变换左上角和右下角象限
	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	//变换右上角和左下角象限
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	//归一化处理，用0-1之间的浮点数将矩阵变换为可视的图像格式


	//////!!!!![神坑]????/////此处得归一化到0-255，不然Qimage不行，而0-1cv它行
	normalize(magI, magI, 0, 255, NORM_MINMAX);

	cv::resize(magI, magI, Size(220, 250));

	//////！！！！！此处重点，原本32F得转化成8U//////32Fto8U
	magI.convertTo(magI, CV_8UC1);

	QImage img1 = cvGrayMat2QImage(magI);
	label = new QLabel();
	label->setPixmap(QPixmap::fromImage(img1));
	label->resize(QSize(img1.width(), img1.height()));
	ui.scrollArea_5->setWidget(label);
}

//HSI
void QtGuiDIP::on_HSI_clicked() {
	Mat img_RGB, img_HSI;
	//对原始图像操作
	img_RGB = image;
	//新建个HSI图像
	img_HSI.create(img_RGB.rows, img_RGB.cols, CV_8UC3);
	//RGB转HSI
	RGB2HSI(img_RGB, img_HSI);

	QImage img1 = QImage((const unsigned char*)(img_HSI.data), img_HSI.cols, img_HSI.rows, QImage::Format_RGB888);
	label = new QLabel();
	label->setPixmap(QPixmap::fromImage(img1));
	label->resize(QSize(img1.width(), img1.height()));
	ui.scrollArea_5->setWidget(label);
	
}

//灰度图Mat转QImage
QImage QtGuiDIP::cvGrayMat2QImage(const cv::Mat& mat){
	QImage image(mat.cols, mat.rows, QImage::Format_Indexed8);        // Set the color table (used to translate colour indexes to qRgb values)        
	image.setColorCount(256);
	for (int i = 0; i < 256; i++)
	{
		image.setColor(i, qRgb(i, i, i));
	}        // Copy input Mat        
	uchar *pSrc = mat.data;
	for (int row = 0; row < mat.rows; row++)
	{
		uchar *pDest = image.scanLine(row);
		memcpy(pDest, pSrc, mat.cols);
		pSrc += mat.step;
	}
	return image;
}

//RGB转HSI
void QtGuiDIP::RGB2HSI(Mat &RGBImg, Mat &hsi)
{
	int row = RGBImg.rows;
	int col = RGBImg.cols;

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float H = 0.0, S = 0.0, I = 0.0;
			float BV = RGBImg.at<Vec3b>(i, j)[0] / 255.0;
			float GV = RGBImg.at<Vec3b>(i, j)[1] / 255.0;
			float RV = RGBImg.at<Vec3b>(i, j)[2] / 255.0;
			//转换公式
			float num = (float)(RV - GV + RV - BV) / 2.0;
			float den = (float)sqrt((double)((RV - GV)*(RV - GV) + (RV - BV)*(GV - BV)));

			if (den != 0) {
				float cosTA = acos(num / den);
				if (BV <= GV) {
					H = cosTA / (CV_PI * 2);
				}
				else {
					H = (2 * CV_PI - cosTA) / (2 * CV_PI);
				}
			}
			else {
				H = 0;
			}
			float minv = min(min(BV, GV), RV);
			den = BV + GV + RV;
			if (den == 0)
				S = 0;
			else
				S = 1 - (float)(3 * minv / den);

			I = (RV + BV + GV) / 3;

			hsi.at<Vec3b>(i, j)[0] = H * 255;
			hsi.at<Vec3b>(i, j)[1] = S * 255;
			hsi.at<Vec3b>(i, j)[2] = I * 255;
		}
	}

}

//Gamma
void QtGuiDIP::on_Gamma_clicked() {
	//////在Last Step框中显示
	QImage imglast = QImage((const unsigned char*)(imageLast.data), imageLast.cols, imageLast.rows, QImage::Format_RGB888);
	labelLast = new QLabel();
	labelLast->setPixmap(QPixmap::fromImage(imglast));
	labelLast->resize(QSize(imglast.width(), imglast.height()));
	ui.scrollArea_3->setWidget(labelLast);
	
	//////算法部分
	QString gammaIn = ui.GammaIn->toPlainText();//获取框中内容
	gamma = gammaIn.toFloat();//转为float型
	
	Mat imageGamma(imageLast.size(), CV_32FC3);	
	for (int i = 0; i < imageLast.rows; i++) {
		for (int j = 0; j < imageLast.cols; j++) {
			imageGamma.at<Vec3f>(i, j)[0] = pow(imageLast.at<Vec3b>(i, j)[0],gamma);
			imageGamma.at<Vec3f>(i, j)[1] = pow(imageLast.at<Vec3b>(i, j)[1],gamma);
			imageGamma.at<Vec3f>(i, j)[2] = pow(imageLast.at<Vec3b>(i, j)[2],gamma);
		} 
	}	
	//归一化到0~255  	
	normalize(imageGamma, imageGamma, 0, 255, NORM_MINMAX);
	//转换成8bit图像显示//imageGamma.convertTo(imageGamma, CV_8UC3);  	
	convertScaleAbs(imageGamma, imageGamma);

	
	//////在Final框中显示
	QImage img = QImage((const unsigned char*)(imageGamma.data), imageGamma.cols, imageGamma.rows, QImage::Format_RGB888);
	label_2 = new QLabel();
	label_2->setPixmap(QPixmap::fromImage(img));
	label_2->resize(QSize(img.width(), img.height()));
	ui.scrollArea_2->setWidget(label_2);
    imageLast = imageGamma;
}

//Gaussian
void QtGuiDIP::on_Gaussian_clicked() {
	//////上一步框
	QImage imglast = QImage((const unsigned char*)(imageLast.data), imageLast.cols, imageLast.rows, QImage::Format_RGB888);
	labelLast = new QLabel();
	labelLast->setPixmap(QPixmap::fromImage(imglast));
	labelLast->resize(QSize(imglast.width(), imglast.height()));
	ui.scrollArea_3->setWidget(labelLast);
	
	//////算法内容
	Mat image_Gaussian;
	GaussianBlur(image, image_Gaussian, Size(3, 3), 0, 0);
    imageLast = image_Gaussian;

	//////在Final框中显示
	QImage img = QImage((const unsigned char*)(image_Gaussian.data), image_Gaussian.cols, image_Gaussian.rows, QImage::Format_RGB888);
	label_2 = new QLabel();
	label_2->setPixmap(QPixmap::fromImage(img));
	label_2->resize(QSize(img.width(), img.height()));
	ui.scrollArea_2->setWidget(label_2);
	
}

//Mean
void QtGuiDIP::on_Mean_clicked() {
	////上一步框
	QImage imglast = QImage((const unsigned char*)(imageLast.data), imageLast.cols, imageLast.rows, QImage::Format_RGB888);
	labelLast = new QLabel();
	labelLast->setPixmap(QPixmap::fromImage(imglast));
	labelLast->resize(QSize(imglast.width(), imglast.height()));
	ui.scrollArea_3->setWidget(labelLast);

	//算法内容
	Mat image_Mean;
	blur(image, image_Mean, Size(7, 7));
    imageLast = image_Mean;
	//////在Final框中显示
	
	QImage img = QImage((const unsigned char*)(image_Mean.data), image_Mean.cols, image_Mean.rows, QImage::Format_RGB888);
	label_2 = new QLabel();
	label_2->setPixmap(QPixmap::fromImage(img));
	label_2->resize(QSize(img.width(), img.height()));
	ui.scrollArea_2->setWidget(label_2);
}

//Sobel
void QtGuiDIP::on_Sobel_clicked() {
	////上一步框
	QImage imglast = QImage((const unsigned char*)(imageLast.data), imageLast.cols, imageLast.rows, QImage::Format_RGB888);
	labelLast = new QLabel();
	labelLast->setPixmap(QPixmap::fromImage(imglast));
	labelLast->resize(QSize(imglast.width(), imglast.height()));
	ui.scrollArea_3->setWidget(labelLast);

	////算法内容
	Mat image_Sobel;
	Mat grad_x, grad_y;//x方向和y方向的梯度
	Mat abs_grad_x, abs_grad_y;
	//求x方向梯度
	Sobel(image, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	//求y方向梯度
	Sobel(image, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	//梯度合并
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, image_Sobel);

	imageLast = image_Sobel;
	////Final框
	QImage img = QImage((const unsigned char*)(image_Sobel.data), image_Sobel.cols, image_Sobel.rows, QImage::Format_RGB888);
	label_2 = new QLabel();
	label_2->setPixmap(QPixmap::fromImage(img));
	label_2->resize(QSize(img.width(), img.height()));
	ui.scrollArea_2->setWidget(label_2);

}

//Prewitt算子算法
Mat QtGuiDIP::prewitt(Mat &imageP) {
	//cvtColor(imageP, imageP, COLOR_RGB2GRAY);//转灰度图
	//算子模板
	float prewitt_x[9] = {
		-1, 0, 1,
		-1, 0, 1,
		-1, 0, 1
	};
	float prewitt_y[9] = {
		1, 1, 1,
		0, 0, 0,
		-1, -1, -1
	};
	//容器定义
	Mat px = Mat(3, 3, CV_32F, prewitt_x);
	Mat py = Mat(3, 3, CV_32F, prewitt_y);
	Mat dstx = Mat(imageP.size(), imageP.type(), imageP.channels());
	Mat dsty = Mat(imageP.size(), imageP.type(), imageP.channels());
	//Mat dst = Mat(imageP.size(), imageP.type(), imageP.channels());
	//x和y方向滤波
	filter2D(imageP, dstx, imageP.depth(), px);
	filter2D(imageP, dsty, imageP.depth(), py);

	Mat tempx(imageP.size(),CV_32FC3), tempy(imageP.size(), CV_32FC3), temp(imageP.size(), CV_32FC3);
	for (int i = 0; i < imageP.rows; i++) {
		for (int j = 0; j < imageP.cols; j++) {
			tempx.at<Vec3f>(i, j)[0] = dstx.at<Vec3b>(i, j)[0];
			tempx.at<Vec3f>(i, j)[1] = dstx.at<Vec3b>(i, j)[1];
			tempx.at<Vec3f>(i, j)[2] = dstx.at<Vec3b>(i, j)[2];
			tempy.at<Vec3f>(i, j)[0] = dsty.at<Vec3b>(i, j)[0];
			tempy.at<Vec3f>(i, j)[1] = dsty.at<Vec3b>(i, j)[1];
			tempy.at<Vec3f>(i, j)[2] = dsty.at<Vec3b>(i, j)[2];
			temp.at<Vec3f>(i, j)[0] = sqrt(tempx.at<Vec3f>(i, j)[0] * tempx.at<Vec3f>(i, j)[0] +
				tempy.at<Vec3f>(i, j)[0] * tempy.at<Vec3f>(i, j)[0]);
			temp.at<Vec3f>(i, j)[1] = sqrt(tempx.at<Vec3f>(i, j)[1] * tempx.at<Vec3f>(i, j)[1] +
				tempy.at<Vec3f>(i, j)[1] * tempy.at<Vec3f>(i, j)[1]);
			temp.at<Vec3f>(i, j)[2] = sqrt(tempx.at<Vec3f>(i, j)[2] * tempx.at<Vec3f>(i, j)[2] +
				tempy.at<Vec3f>(i, j)[2] * tempy.at<Vec3f>(i, j)[2]);
			//tempx = dstx.at<uchar>(i, j);
			//tempy = dsty.at<uchar>(i, j);
			//temp = sqrt(tempx*tempx + tempy * tempy);
			//dst.at<uchar>(i, j) = temp;
		}
	}
	//return dst;
	//归一化
	normalize(temp, temp, 0, 255, NORM_MINMAX);
	//转成8位图
	convertScaleAbs(temp, temp);
	return temp;
}

//Prewitt
void QtGuiDIP::on_Prewitt_clicked() {
		////上一步框
		QImage imglast = QImage((const unsigned char*)(imageLast.data), imageLast.cols, imageLast.rows, QImage::Format_RGB888);
		labelLast = new QLabel();
		labelLast->setPixmap(QPixmap::fromImage(imglast));
		labelLast->resize(QSize(imglast.width(), imglast.height()));
		ui.scrollArea_3->setWidget(labelLast);

		////算法内容
		Mat image_Prewitt;
		image_Prewitt = prewitt(image);
		//image_Prewitt.convertTo(image_Prewitt, CV_8UC1);
        imageLast = image_Prewitt;
		////Final框
		QImage img = QImage((const unsigned char*)(image_Prewitt.data), image_Prewitt.cols, image_Prewitt.rows, QImage::Format_RGB888);
		//QImage img = cvGrayMat2QImage(image_Prewitt);
		label_2 = new QLabel();
		label_2->setPixmap(QPixmap::fromImage(img));
		label_2->resize(QSize(img.width(), img.height()));
		ui.scrollArea_2->setWidget(label_2);
		//image_Prewitt.convertTo(image_Prewitt, CV_32F);
        
}

//MyLaplacian
Mat QtGuiDIP::myLaplacian(Mat &imageL) {
	float myLap[9] = {
		0, 1, 0,
		1, -4, 1,
		0, 1, 0
	};
	Mat image_Lap;
	Mat Lap = Mat(3, 3, CV_32F, myLap);
	Mat dst = Mat(imageL.size(), imageL.type(), imageL.channels());
	filter2D(imageL, dst, imageL.depth(), Lap);
	Mat temp(imageL.size(), CV_32FC3);
	for (int i = 0; i < imageL.rows; i++) {
		for (int j = 0; j < imageL.cols; j++) {
			temp.at<Vec3f>(i, j)[0] = dst.at<Vec3b>(i, j)[0];
			temp.at<Vec3f>(i, j)[1] = dst.at<Vec3b>(i, j)[1];
			temp.at<Vec3f>(i, j)[2] = dst.at<Vec3b>(i, j)[2];
		}
	}
	normalize(temp, temp, 0, 255, NORM_MINMAX);
	//转成8位图
	convertScaleAbs(temp, temp);
	return temp;
}

//Laplace
void QtGuiDIP::on_Laplacian_clicked(){
	QImage imglast = QImage((const unsigned char*)(imageLast.data), imageLast.cols, imageLast.rows, QImage::Format_RGB888);
	labelLast = new QLabel();
	labelLast->setPixmap(QPixmap::fromImage(imglast));
	labelLast->resize(QSize(imglast.width(), imglast.height()));
	ui.scrollArea_3->setWidget(labelLast);

	////算法内容
	Mat image_Laplacian;
	image_Laplacian = myLaplacian(image);

	imageLast = image_Laplacian;

	////Final框
	QImage img = QImage((const unsigned char*)(image_Laplacian.data), image_Laplacian.cols, image_Laplacian.rows, QImage::Format_RGB888);
	label_2 = new QLabel();
	label_2->setPixmap(QPixmap::fromImage(img));
	label_2->resize(QSize(img.width(), img.height()));
	ui.scrollArea_2->setWidget(label_2);
}

//保存文件
void QtGuiDIP::on_Save_clicked() {
	Mat imageSave;
	imageSave = imageLast;
	//电脑显示图片为BGR
	cvtColor(imageSave, imageSave, COLOR_RGB2BGR);

	QString fileName;
	fileName = QFileDialog::getSaveFileName(this,
		tr("Save as"),
		"",
		tr("Images(*.png)"));
	string fileAsSave = fileName.toStdString();
	imwrite(fileAsSave, imageSave);
}

//缩放
void QtGuiDIP::on_Scale_clicked() {
	//////在Last Step框中显示
	QImage imglast = QImage((const unsigned char*)(imageLast.data), imageLast.cols, imageLast.rows, QImage::Format_RGB888);
	labelLast = new QLabel();
	labelLast->setPixmap(QPixmap::fromImage(imglast));
	labelLast->resize(QSize(imglast.width(), imglast.height()));
	ui.scrollArea_3->setWidget(labelLast);

	//////算法部分
	QString scaleIn = ui.ScaleIn->toPlainText();//获取框中内容
	scale = scaleIn.toFloat();//转为float型
	Mat image_Scale;
	image_Scale = imageLast;
	cv::resize(image_Scale, image_Scale, Size(220 * scale, 250 * scale));

	/////在Final框中显示
	QImage img = QImage((const unsigned char*)(image_Scale.data), image_Scale.cols, image_Scale.rows, QImage::Format_RGB888);
	label_2 = new QLabel();
	label_2->setPixmap(QPixmap::fromImage(img));
	label_2->resize(QSize(img.width(), img.height()));
	ui.scrollArea_2->setWidget(label_2);

}

//RGB转gray
void QtGuiDIP::on_ToGray_clicked() {
	//////显示到上一步
	QImage imglast = QImage((const unsigned char*)(imageLast.data), imageLast.cols, imageLast.rows, QImage::Format_RGB888);
	labelLast = new QLabel();
	labelLast->setPixmap(QPixmap::fromImage(imglast));
	labelLast->resize(QSize(imglast.width(), imglast.height()));
	ui.scrollArea_3->setWidget(labelLast);

	/////
	Mat image_Gray;
	cvtColor(image, image_Gray, COLOR_RGB2GRAY);
	imageLastC1 = image_Gray;

	QImage img1 = cvGrayMat2QImage(image_Gray);
	label = new QLabel();
	label->setPixmap(QPixmap::fromImage(img1));
	label->resize(QSize(img1.width(), img1.height()));
	ui.scrollArea_2->setWidget(label);

}

//OTSU二值化
void QtGuiDIP::on_OTSU_clicked() {
	//////显示到上一步
	QImage imglast = cvGrayMat2QImage(imageLastC1);
	labelLast = new QLabel();
	labelLast->setPixmap(QPixmap::fromImage(imglast));
	labelLast->resize(QSize(imglast.width(), imglast.height()));
	ui.scrollArea_3->setWidget(labelLast);

	//////
	Mat image_OTSU;
	//image_OTSU = imageLastC1;
	threshold(imageLastC1, image_OTSU, 0, 255, THRESH_OTSU);
	imageLastC1 = image_OTSU;
	
	//////
	QImage img1 = cvGrayMat2QImage(image_OTSU);
	label = new QLabel();
	label->setPixmap(QPixmap::fromImage(img1));
	label->resize(QSize(img1.width(), img1.height()));
	ui.scrollArea_2->setWidget(label);
}

//膨胀
void QtGuiDIP::on_Dilate_clicked() {
	//////显示到上一步
	QImage imglast = cvGrayMat2QImage(imageLastC1);
	labelLast = new QLabel();
	labelLast->setPixmap(QPixmap::fromImage(imglast));
	labelLast->resize(QSize(imglast.width(), imglast.height()));
	ui.scrollArea_3->setWidget(labelLast);

	//////算法
	//自定义核
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat image_Dilate;
	//膨胀
	dilate(imageLastC1, image_Dilate, element);
	imageLastC1 = image_Dilate;

	//////
	QImage img1 = cvGrayMat2QImage(image_Dilate);
	label = new QLabel();
	label->setPixmap(QPixmap::fromImage(img1));
	label->resize(QSize(img1.width(), img1.height()));
	ui.scrollArea_2->setWidget(label);

}

//腐蚀
void QtGuiDIP::on_Erode_clicked() {
	//////显示到上一步
	QImage imglast = cvGrayMat2QImage(imageLastC1);
	labelLast = new QLabel();
	labelLast->setPixmap(QPixmap::fromImage(imglast));
	labelLast->resize(QSize(imglast.width(), imglast.height()));
	ui.scrollArea_3->setWidget(labelLast);

	//////算法
	//自定义核
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat image_Erode;
	//腐蚀
	erode(imageLastC1, image_Erode, element);
	imageLastC1 = image_Erode;

	//////
	QImage img1 = cvGrayMat2QImage(image_Erode);
	label = new QLabel();
	label->setPixmap(QPixmap::fromImage(img1));
	label->resize(QSize(img1.width(), img1.height()));
	ui.scrollArea_2->setWidget(label);
}

//开运算 先腐蚀后膨胀
void QtGuiDIP::on_OpenOperate_clicked() {
	//////显示到上一步
	QImage imglast = cvGrayMat2QImage(imageLastC1);
	labelLast = new QLabel();
	labelLast->setPixmap(QPixmap::fromImage(imglast));
	labelLast->resize(QSize(imglast.width(), imglast.height()));
	ui.scrollArea_3->setWidget(labelLast);

	/////
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat image_Open;
	morphologyEx(imageLastC1, image_Open, MORPH_OPEN, element);
	imageLastC1 = image_Open;

	//////
	QImage img1 = cvGrayMat2QImage(image_Open);
	label = new QLabel();
	label->setPixmap(QPixmap::fromImage(img1));
	label->resize(QSize(img1.width(), img1.height()));
	ui.scrollArea_2->setWidget(label);
}

//闭运算
void QtGuiDIP::on_CloseOperate_clicked() {
	//////显示到上一步
	QImage imglast = cvGrayMat2QImage(imageLastC1);
	labelLast = new QLabel();
	labelLast->setPixmap(QPixmap::fromImage(imglast));
	labelLast->resize(QSize(imglast.width(), imglast.height()));
	ui.scrollArea_3->setWidget(labelLast);

	//////
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat image_Close;
	morphologyEx(imageLastC1, image_Close, MORPH_CLOSE, element);
	imageLastC1 = image_Close;

	//////
	QImage img1 = cvGrayMat2QImage(image_Close);
	label = new QLabel();
	label->setPixmap(QPixmap::fromImage(img1));
	label->resize(QSize(img1.width(), img1.height()));
	ui.scrollArea_2->setWidget(label);
}