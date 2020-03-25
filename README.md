# OpenCV_translate_feature2D
this repository will show the  translation of opencv official word  that including code in three languages c++ python and java
# Harris 角点检测
在本教程中，您将学习：</br>
有哪些特征及其重要性。</br>
使用函数cv：：cornerHarris使用Harris-Stephens方法检测角点。</br>
## 理论
### 什么是特征？
&emsp;在计算机视觉中，通常需要在环境的不同帧之间找到匹配点。为什么？如果我们知道两个图像是如何相互关联的，我们可以使用这两个图像来提取它们的信息。</br>
&emsp;当我们说匹配点时，一般来说，我们指的是场景中容易识别的特征。我们称这些特征为特征。</br>
&emsp;那么，一个特征应该具有哪些特征呢？它必须是唯一可识别的。</br>
## 图像特征类型
举几个例子：</br>
&emsp;边缘</br>
&emsp;角点（也称为兴趣点）</br>
&emsp;Blobs（也称为感兴趣的区域）</br>
在本教程中，我们将特别研究角点特征。
## 为什么角点这么特别？
&emsp;因为它是两条边的交集,它表示这两条边的方向改变的点。因此，图像的梯度（在两个方向上）有很大的变化，可以用来检测它。
## 它是如何工作的？
我们寻找一个角点，由于角点代表图像中渐变的变化，我们将寻找这种“变化”。</br>
考虑一个*灰度图像*$\(I)$。我们将扫描一个窗口$\w（x，y）$,（位移u在x方向，v在y方向）$\I$并计算强度的变化。</br>
质能守恒方程可以用一个很简洁的方程式 $E=mc^2$ 来表达。
```
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;
Mat src, src_gray;
int thresh = 200;
int max_thresh = 255;
const char* source_window = "Source image";
const char* corners_window = "Corners detected";
void cornerHarris_demo(int, void*);
int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, "{@input | building.jpg | input image}");
    src = imread(samples::findFile(parser.get<String>("@input")));
    if (src.empty())
    {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    namedWindow(source_window);
    createTrackbar("Threshold: ", source_window, &thresh, max_thresh, cornerHarris_demo);
    imshow(source_window, src);
    cornerHarris_demo(0, 0);
    waitKey();
    return 0;
}
void cornerHarris_demo(int, void*)
{
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    Mat dst = Mat::zeros(src.size(), CV_32FC1);
    cornerHarris(src_gray, dst, blockSize, apertureSize, k);
    Mat dst_norm, dst_norm_scaled;
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(dst_norm, dst_norm_scaled);
    for (int i = 0; i < dst_norm.rows; i++)
    {
        for (int j = 0; j < dst_norm.cols; j++)
        {
            if ((int)dst_norm.at<float>(i, j) > thresh)
            {
                circle(dst_norm_scaled, Point(j, i), 5, Scalar(0), 2, 8, 0);
            }
        }
    }
    namedWindow(corners_window);
    imshow(corners_window, dst_norm_scaled);
}


