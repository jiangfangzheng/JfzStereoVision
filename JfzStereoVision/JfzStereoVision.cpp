#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define CAML 1     //左右摄像头载入编号，自己调一下
#define CAMR 0
#define WIDTH  352 //摄像头分辨率设置
#define HEIGHT 288

int cnt = 1;//拍照图片计数
char Leftname[25];//生成图像文件名
char Rightname[25];

CvCapture *cap1 = NULL, *cap2 = NULL; //capture1为left, capture2为right
Mat disp, disp8u, pointClouds, imageLeft, imageRight, disparityImage;
Mat depth;//分离出的深度Mat
bool left_mouse = false;//鼠标左键按下标志位

bool    m_Calib_Data_Loaded;        // 是否成功载入定标参数
Mat m_Calib_Mat_Q;              // Q 矩阵
Mat m_Calib_Mat_Remap_X_L;      // 左视图畸变校正像素坐标映射矩阵 X
Mat m_Calib_Mat_Remap_Y_L;      // 左视图畸变校正像素坐标映射矩阵 Y
Mat m_Calib_Mat_Remap_X_R;      // 右视图畸变校正像素坐标映射矩阵 X
Mat m_Calib_Mat_Remap_Y_R;      // 右视图畸变校正像素坐标映射矩阵 Y
Mat m_Calib_Mat_Mask_Roi;       // 左视图校正后的有效区域
Rect m_Calib_Roi_L;             // 左视图校正后的有效区域矩形
Rect m_Calib_Roi_R;             // 右视图校正后的有效区域矩形

double          m_FL;//某参数
int pic_info[2];//存测距时x、y像素坐标

StereoBM    m_BM;
int m_numberOfDisparies;            // 视差变化范围，深度图伪彩色化时要用

//算法参数、轨迹条调节参数
int SGBM_SADWindowSize = 10;
int SGBM_numberOfDisparities = 64;
int SGBM_uniquenessRatio = 15;

//极简单的输出某个pixel（像素）所对应的三维的坐标，像素点转坐标，Z轴显示深度,x、y小于图片长宽，从0开始
void PixelToAxis(Mat xyz, int x, int y)
{
	Point p;
	p.x = x; p.y = y;//小于图片长宽
	//cout << x << "," << y << " 世界坐标: " << xyz.at<Vec3f>(p) << endl;
	/**********OpenCV数据转实际距离处理方法**********/
	// 提取深度图像
	vector<Mat> xyzSet;
	split(xyz, xyzSet);//xyz三通道分离
	xyzSet[2].copyTo(depth);//第三个通道即为深度图
	float distance;
	distance = depth.at<float>(p);//distance即等于xyz.at<Vec3f>(p)操作中得到的z值
	// 线性回归的方式算映射实际距离
	distance = distance*(-1000);//扩大-1000倍
	distance = 0.4164*distance + (-9.2568);
	cout << x << "," << y << " 距离: " << distance << " cm" << endl;
}

static void onMouse(int event, int x, int y, int /*flags*/, void* /*param*/)
{
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		pic_info[0] = x;
		pic_info[1] = y;
		PixelToAxis(pointClouds, x, y);//像素转坐标
		left_mouse = true;
	}
	else if (event == CV_EVENT_LBUTTONUP)
	{
		left_mouse = false;
	}
	else if ((event == CV_EVENT_MOUSEMOVE) && (left_mouse == true))
	{
	}
}

//CreateTrackbar的回调函数
int SADWindowSizeValue = 10; //SADWindowSizeValue值
static void SADWindowSizeControl(int, void *)
{
	if (SADWindowSizeValue < 5)
	{
		SADWindowSizeValue = 5;
		SGBM_SADWindowSize = SADWindowSizeValue;
	}
	else if (SADWindowSizeValue % 2 == 0)
	{
		SADWindowSizeValue += 1;
		SGBM_SADWindowSize = SADWindowSizeValue;
	}
	else
	{
		SGBM_SADWindowSize = SADWindowSizeValue;
	}
}

int numberOfDisparitiesValue = 64; //numberOfDisparities值
static void numberOfDisparitiesControl(int, void *)
{
	while (numberOfDisparitiesValue % 16 != 0 || numberOfDisparitiesValue == 0)
	{
		numberOfDisparitiesValue++;
	}
	SGBM_numberOfDisparities = numberOfDisparitiesValue;
}

int uniquenessRatioValue = 15; //uniquenessRatio值
static void uniquenessRatioControl(int, void *)
{

	SGBM_uniquenessRatio = uniquenessRatioValue;
}


int loadCalibData()
{
	// 读入摄像头定标参数 Q roi1 roi2 mapx1 mapy1 mapx2 mapy2
	try
	{
		cv::FileStorage fs("calib_paras.xml", cv::FileStorage::READ);
		cout << fs.isOpened() << endl;

		if (!fs.isOpened())
		{
			return (0);
		}

		cv::Size imageSize;
		cv::FileNodeIterator it = fs["imageSize"].begin();

		it >> imageSize.width >> imageSize.height;

		vector<int> roiVal1;
		vector<int> roiVal2;

		fs["leftValidArea"] >> roiVal1;

		m_Calib_Roi_L.x = roiVal1[0];
		m_Calib_Roi_L.y = roiVal1[1];
		m_Calib_Roi_L.width = roiVal1[2];
		m_Calib_Roi_L.height = roiVal1[3];

		fs["rightValidArea"] >> roiVal2;
		m_Calib_Roi_R.x = roiVal2[0];
		m_Calib_Roi_R.y = roiVal2[1];
		m_Calib_Roi_R.width = roiVal2[2];
		m_Calib_Roi_R.height = roiVal2[3];


		fs["QMatrix"] >> m_Calib_Mat_Q;
		fs["remapX1"] >> m_Calib_Mat_Remap_X_L;
		fs["remapY1"] >> m_Calib_Mat_Remap_Y_L;
		fs["remapX2"] >> m_Calib_Mat_Remap_X_R;
		fs["remapY2"] >> m_Calib_Mat_Remap_Y_R;

		cv::Mat lfCamMat;
		fs["leftCameraMatrix"] >> lfCamMat;
		m_FL = lfCamMat.at<double>(0, 0);

		m_Calib_Mat_Q.at<double>(3, 2) = -m_Calib_Mat_Q.at<double>(3, 2);

		m_Calib_Mat_Mask_Roi = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC1);
		cv::rectangle(m_Calib_Mat_Mask_Roi, m_Calib_Roi_L, cv::Scalar(255), -1);

		m_BM.state->roi1 = m_Calib_Roi_L;
		m_BM.state->roi2 = m_Calib_Roi_R;

		m_Calib_Data_Loaded = true;

		string method;
		fs["rectifyMethod"] >> method;
		if (method != "BOUGUET")
		{
			return (-2);
		}

	}
	catch (std::exception& e)
	{
		m_Calib_Data_Loaded = false;
		return (-99);
	}

	return 1;


}

int getDisparityImage(cv::Mat& disparity, cv::Mat& disparityImage, bool isColor)
{
	// 将原始视差数据的位深转换为 8 位
	cv::Mat disp8u;
	if (disparity.depth() != CV_8U)
	{
		if (disparity.depth() == CV_8S)
		{
			disparity.convertTo(disp8u, CV_8U);
		}
		else
		{
			disparity.convertTo(disp8u, CV_8U, 255 / (m_numberOfDisparies*16.));
		}
	}
	else
	{
		disp8u = disparity;
	}

	// 转换为伪彩色图像 或 灰度图像
	if (isColor)
	{
		if (disparityImage.empty() || disparityImage.type() != CV_8UC3 || disparityImage.size() != disparity.size())
		{
			disparityImage = cv::Mat::zeros(disparity.rows, disparity.cols, CV_8UC3);
		}

		for (int y = 0; y < disparity.rows; y++)
		{
			for (int x = 0; x < disparity.cols; x++)
			{
				uchar val = disp8u.at<uchar>(y, x);
				uchar r, g, b;

				if (val == 0)
					r = g = b = 0;
				else
				{
					r = 255 - val;
					g = val < 128 ? val * 2 : (uchar)((255 - val) * 2);
					b = val;
				}

				disparityImage.at<cv::Vec3b>(y, x) = cv::Vec3b(r, g, b);

			}
		}
	}
	else
	{
		disp8u.copyTo(disparityImage);
	}

	return 1;
}

Mat Match_BM(Mat left, Mat right, Rect roi1, Rect roi2)//BM匹配算法，输入left、right为灰度图,roi1、roi2为左右视图的有效像素区域，一般由双目校正阶段的cvStereoRectify 函数传递，输出disp8为灰度图
{
	StereoBM bm;
	int SADWindowSize = 21;//主要影响参数
	int numberOfDisparities = 64;//主要影响参数
	int uniquenessRatio = 3;//主要影响参数
	bm.state->roi1 = roi1;
	bm.state->roi2 = roi2;
	bm.state->preFilterCap = 63;
	bm.state->SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 9;
	bm.state->minDisparity = 0;
	bm.state->numberOfDisparities = numberOfDisparities;
	bm.state->textureThreshold = 29;
	bm.state->uniquenessRatio = uniquenessRatio;
	bm.state->speckleWindowSize = 200;
	bm.state->speckleRange = 32;
	bm.state->disp12MaxDiff = 2;
	Mat disp, disp8;
	int64 t = getTickCount();
	bm(left, right, disp);
	t = getTickCount() - t;
	cout << "BM耗时:" << t * 1000 / getTickFrequency() << "毫秒，即" << t / getTickFrequency() << "秒" << endl;
	disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities*16.));
	return disp8;
}

Mat Match_SGBM(Mat left, Mat right)//SGBM匹配算法，输入left、right为灰度图，输出disp8为灰度图
{
	StereoSGBM sgbm;
	int SADWindowSize = SGBM_SADWindowSize;//主要影响参数
	int numberOfDisparities = SGBM_numberOfDisparities;//主要影响参数
	int uniquenessRatio = SGBM_uniquenessRatio;//主要影响参数，越大误匹配越小，不能匹配区域越多
	int cn = left.channels();
	sgbm.preFilterCap = 63;
	sgbm.SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 3;
	sgbm.P1 = 8 * cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.P2 = 32 * cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.minDisparity = 0;
	sgbm.numberOfDisparities = numberOfDisparities;
	sgbm.uniquenessRatio = uniquenessRatio;
	sgbm.speckleWindowSize = 100;
	sgbm.speckleRange = 32;
	sgbm.disp12MaxDiff = 1;
	Mat disp, disp8;
	int64 t = getTickCount();
	sgbm(left, right, disp);
	t = getTickCount() - t;
	cout << "SGBM耗时:" << t * 1000 / getTickFrequency() << "毫秒，即" << t / getTickFrequency() << "秒" << endl;
	disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities*16.));
	return disp8;
}

Mat Match_Var(Mat left, Mat right)//Var立体匹配算法(OpenCV2.3新增)，输入left、right为灰度图，输出disp8为灰度图
{
	StereoVar var;
	int numberOfDisparities = 128;//影响参数64
	var.levels = 3;//忽略则使用自动参数(USE_AUTO_PARAMS)
	var.pyrScale = 0.5;//忽略则使用自动参数(USE_AUTO_PARAMS)
	var.nIt = 25;
	var.minDisp = -numberOfDisparities;
	var.maxDisp = 0;
	var.poly_n = 3;
	var.poly_sigma = 0.0;
	var.fi = 15.0f;
	var.lambda = 0.03f;
	var.penalization = var.PENALIZATION_TICHONOV; //忽略则使用自动参数(USE_AUTO_PARAMS)
	var.cycle = var.CYCLE_V;// 忽略则使用自动参数(USE_AUTO_PARAMS)
	var.flags = var.USE_SMART_ID | var.USE_AUTO_PARAMS | var.USE_INITIAL_DISPARITY | var.USE_MEDIAN_FILTERING;
	Mat disp, disp8;
	int64 t = getTickCount();
	var(left, right, disp);
	t = getTickCount() - t;
	cout << "Var耗时:" << t * 1000 / getTickFrequency() << "毫秒，即" << t / getTickFrequency() << "秒" << endl;
	disp.convertTo(disp8, CV_8U);//注意Var算法与其它的这里不一样
	return disp8;
}

void updatebm()
{
	m_BM.state->preFilterSize = 63;
	m_BM.state->preFilterCap = 63;
	m_BM.state->SADWindowSize = 21;
	m_BM.state->minDisparity = 0;
	m_BM.state->numberOfDisparities = 64;
	m_BM.state->textureThreshold = 29;
	m_BM.state->uniquenessRatio = 3;
	m_BM.state->speckleWindowSize = 200;
	m_BM.state->speckleRange = 32;
	m_BM.state->disp12MaxDiff = 2;

}

int  bmMatch(cv::Mat& frameLeft, cv::Mat& frameRight, cv::Mat& disparity, cv::Mat& imageLeft, cv::Mat& imageRight)
{
	Mat Left_processing, Right_processing;
	// 输入检查
	if (frameLeft.empty() || frameRight.empty())
	{
		disparity = cv::Scalar(0);
		return 0;
	}
	if (WIDTH == 0 || HEIGHT == 0)
	{
		return 0;
	}

	// 转换为灰度图
	cv::Mat img1proc, img2proc;
	cvtColor(frameLeft, img1proc, CV_BGR2GRAY);
	cvtColor(frameRight, img2proc, CV_BGR2GRAY);

	// 校正图像，使左右视图行对齐    
	cv::Mat img1remap, img2remap;

	if (m_Calib_Data_Loaded)
	{
		remap(img1proc, img1remap, m_Calib_Mat_Remap_X_L, m_Calib_Mat_Remap_Y_L, cv::INTER_LINEAR);     // 对用于视差计算的画面进行校正
		remap(img2proc, img2remap, m_Calib_Mat_Remap_X_R, m_Calib_Mat_Remap_Y_R, cv::INTER_LINEAR);
	}
	else
	{
		img1remap = img1proc;
		img2remap = img2proc;
	}

	// 对左右视图的左边进行边界延拓，以获取与原始视图相同大小的有效视差区域
	cv::Mat img1border, img2border;
	if (m_numberOfDisparies != m_BM.state->numberOfDisparities)
		m_numberOfDisparies = m_BM.state->numberOfDisparities;
	copyMakeBorder(img1remap, img1border, 0, 0, m_BM.state->numberOfDisparities, 0, IPL_BORDER_REPLICATE);
	copyMakeBorder(img2remap, img2border, 0, 0, m_BM.state->numberOfDisparities, 0, IPL_BORDER_REPLICATE);

	//计算矫正后图片的有效部分、左右共有部分
	Rect vroiOK[3];//存有效区域矩形，用于求交集
	vroiOK[2] = m_Calib_Roi_L & m_Calib_Roi_R;//vroiOK[2]为交集

	if (m_Calib_Data_Loaded)
	{
		remap(frameLeft, Left_processing, m_Calib_Mat_Remap_X_L, m_Calib_Mat_Remap_Y_L, cv::INTER_LINEAR);
		//rectangle(Left_processing, vroiOK[2], CV_RGB(0, 0, 255), 3);//蓝色框画出共有或有效部分
	}
	else
		frameLeft.copyTo(Left_processing);


	if (m_Calib_Data_Loaded)
	{ 
		remap(frameRight, Right_processing, m_Calib_Mat_Remap_X_R, m_Calib_Mat_Remap_Y_R, cv::INTER_LINEAR);
		//rectangle(Right_processing, vroiOK[2], CV_RGB(0, 0, 255), 3);//蓝色框画出共有或有效部分
	}
	else
		frameRight.copyTo(Right_processing);
	
	// 左右图共有部分剪裁下来
	imageLeft = Left_processing(vroiOK[2]);//left中感兴趣部分(vroiOK[2]矩形的区域)赋给imageROI
	imageRight = Right_processing(vroiOK[2]);//left中感兴趣部分(vroiOK[2]矩形的区域)赋给imageROI

	// 计算视差
	Mat imageLeftGRAY, imageRihtGRAY;
	cvtColor(imageLeft, imageLeftGRAY, CV_BGR2GRAY);//左右图灰度转彩色
	cvtColor(imageRight, imageRihtGRAY, CV_BGR2GRAY);//左右图灰度转彩

	////BM算法计算视差
	//Mat BMdisparity, BMdisparity2;
	//m_BM(imageLeftGRAY, imageRihtGRAY, BMdisparity);
	//getDisparityImage(BMdisparity, BMdisparity2, true);
	//imshow("BM算法", BMdisparity2);

	//Rect roi1, roi2;
	//Mat BMdisparity3, BMdisparity4;
	//BMdisparity3 = Match_BM(imageLeftGRAY, imageRihtGRAY, roi1, roi2);
	//getDisparityImage(BMdisparity3, BMdisparity4, true);
	//imshow("BM cpp算法", BMdisparity4);

	//SGBM算法计算视差
	disparity = Match_SGBM(imageLeftGRAY, imageRihtGRAY);//SGBM匹配算法，输入left、right为灰度图，输出disp8为灰度图

	////Var算法计算视差
	//Mat Vardisparity, Vardisparity2;
	//Vardisparity = Match_Var(imageLeftGRAY, imageRihtGRAY);
	//getDisparityImage(Vardisparity, Vardisparity2, true);
	//imshow("Var算法", Vardisparity2);

	return 1;
}



int getPointClouds(cv::Mat& disparity, cv::Mat& pointClouds)
{
	if (disparity.empty())
	{
		return 0;
	}
	reprojectImageTo3D(disparity, pointClouds, m_Calib_Mat_Q, true);//计算生成三维点云
	pointClouds *= 1.6;//点云数据扩大1.6倍
	return 1;
}



int main(int argc, char** argv)
{
	//载入摄像头
	cap1 = cvCaptureFromCAM(CAML);
	assert(cap1 != NULL);
	cvWaitKey(100);
	cap2 = cvCaptureFromCAM(CAMR);
	assert(cap2 != NULL);
	//设置画面尺寸 WIDTH、HEIGHT在宏定义中改
	cvSetCaptureProperty(cap1, CV_CAP_PROP_FRAME_WIDTH, WIDTH);
	cvSetCaptureProperty(cap1, CV_CAP_PROP_FRAME_HEIGHT, HEIGHT);
	cvSetCaptureProperty(cap2, CV_CAP_PROP_FRAME_WIDTH, WIDTH);
	cvSetCaptureProperty(cap2, CV_CAP_PROP_FRAME_HEIGHT, HEIGHT);
	//建立显示窗口
	namedWindow("左Left", 1);
	namedWindow("右Right", 1);
	namedWindow("视差图", 1);
	//鼠标点击回显
	setMouseCallback("视差图", onMouse, 0);
	//载入摄像头内外参
	loadCalibData();
	cout << "双目标定参数载入成功！" << endl;

	/*轨迹条使用*/
	namedWindow("【SGBM调整窗口】", 1);
	Mat Logo = imread("Logo.png", 1);
	imshow("【SGBM调整窗口】", Logo);
	createTrackbar("SADWindowSize：", "【SGBM调整窗口】", &SADWindowSizeValue, 25, SADWindowSizeControl);//调的参数名、窗口名、载入值、最大值、回调函数
	SADWindowSizeControl(SADWindowSizeValue, 0);//调用回调函数 
	createTrackbar("numberOfDisparities：", "【SGBM调整窗口】", &numberOfDisparitiesValue, 256, numberOfDisparitiesControl);
	numberOfDisparitiesControl(numberOfDisparitiesValue, 0);
	createTrackbar("uniquenessRatio：", "【SGBM调整窗口】", &uniquenessRatioValue, 50, uniquenessRatioControl);
	uniquenessRatioControl(uniquenessRatioValue, 0);


	while (true)
	{
		Mat frame1;
		Mat frame2;
		frame1 = cvQueryFrame(cap1);
		frame2 = cvQueryFrame(cap2);
		if (frame1.empty())   break;
		if (frame2.empty())   break;
		updatebm();//匹配算法参数更新
		bmMatch(frame1, frame2, disp, imageLeft, imageRight);//矫正+匹配
		imshow("左Left", imageLeft);
		imshow("右Right", imageRight);
		getDisparityImage(disp, disparityImage, true);//true时，disp转彩色，否则为灰度图
		getPointClouds(disp, pointClouds);//视差图3D点云重建
		imshow("视差图", disparityImage);

		char c = waitKey(100);
		if (c == '1') //按'1'保存左右矫正好的照片
		{
			sprintf(Leftname, "%02dLeft-Rectified.png", cnt);
			sprintf(Rightname, "%02dRight-Rectified.png", cnt);
			imwrite(Leftname, imageLeft);
			imwrite(Rightname, imageRight);
			cnt++;
			cout << Leftname << " 和 " << Rightname << " 保存成功！" << endl;
		}
	}
	return 0;
}

