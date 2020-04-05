## 数字图像处理系统

系统架构以及界面展示详见“系统架构及Qt界面展示.pdf”，源代码位于DIP文件夹下，需配置Opencv。

1. 功能简介：
   + Flip左右翻转、Scale图像缩放（输入小数）
   + Gamma点运算（输入小数）       
   + Gaussian 、Mean 模板运算（图像平滑模糊处理）
   + Laplacian、Sobel、Prewitt 边缘检测
   + OTSU 图像二值化（大津）
   + FFT傅里叶变换  || HSI颜色空间 || color hist 颜色直方图
   + Dilate膨胀、Ecrode腐蚀、OpenOperate开运算、CloseOperate闭运算（形态学）
2. 注意：
   + Gaussian、Mean只对原图进行操作
   + FFT、HSI、颜色直方图只对原图进行操作
   + 右边的ToGray可相当于灰度图初始化，二值化以及形态学的操作不和左边的彩色图像的操作交叉使用