# JfzStereoVision
基于OpenCV的双目视觉匹配测距系统

软件有两个工程：

JfzStereoImgGet：用于拍照，拍摄左右摄像头的图片，图片用于棋盘格标定。

JfzStereoVision：用于载入相机参数，从而进行匹配测距，生成相机参数的模块没做，直接用了邹宇华的上位机进行生成，在此非常感谢邹老师。