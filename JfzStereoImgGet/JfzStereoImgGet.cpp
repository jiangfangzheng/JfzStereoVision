#include <iostream>
#include "afx.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

//左右摄像头载入编号，自己调一下
#define CAML 1
#define CAMR 2

//摄像头尺寸设置
#define WIDTH 352
#define HEIGHT 288

CvCapture *capture1 = NULL, *capture2 = NULL; //capture1为left, capture2为right
Mat  imageLeft;
Mat  imageRight;

int cnt = 1;//拍照图片计数
char Leftname[25];//生成图像文件名
char Rightname[25];

int main(int argc, char *argv[])
{
	//载入摄像头
	capture1 = cvCaptureFromCAM(CAML);
	assert(capture1 != NULL);
	cvWaitKey(100);
	capture2 = cvCaptureFromCAM(CAMR);
	assert(capture2 != NULL);

	//设置画面尺寸 WIDTH、HEIGHT在宏定义中改
	cvSetCaptureProperty(capture1, CV_CAP_PROP_FRAME_WIDTH, WIDTH);
	cvSetCaptureProperty(capture1, CV_CAP_PROP_FRAME_HEIGHT, HEIGHT);
	cvSetCaptureProperty(capture2, CV_CAP_PROP_FRAME_WIDTH, WIDTH);
	cvSetCaptureProperty(capture2, CV_CAP_PROP_FRAME_HEIGHT, HEIGHT);

	cout << "视频分辨率设置为：" << WIDTH << " x " << HEIGHT << endl;

	imageLeft = cvQueryFrame(capture1);
	imageRight = cvQueryFrame(capture2);

	while (true)
	{ 
		imageLeft = cvQueryFrame(capture1);
		imageRight = cvQueryFrame(capture2);
		imshow("Left", imageLeft);
		imshow("Right", imageRight);
		char c = waitKey(100);
		if (c == '1') //按'g'保存照片
		{
			sprintf(Leftname, "stereoData\\Left%02d.jpg", cnt);
			sprintf(Rightname, "stereoData\\Right%02d.jpg", cnt);
			imwrite(Leftname, imageLeft);
			imwrite(Rightname, imageRight);
			cnt++;
			cout << Leftname <<" 和 "<<Rightname << " 保存成功！" << endl;
		}
	}
	char c = waitKey();
	if (c == 27) //按ESC键退出
		return 0;
}
