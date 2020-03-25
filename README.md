# OpenCV_translate_feature2D
this repository will show the  translation of opencv official word  that including code in three languages c++ python and java
<!-- more -->
# Harris 角点检测
在本教程中，您将学习：
有哪些特征及其重要性。
使用函数cv::cornerHarris使用Harris-Stephens方法检测角点。
## 理论
### 什么是特征？
&emsp;在计算机视觉中，通常需要在环境的不同帧之间找到匹配点。为什么？如果我们知道两个图像是如何相互关联的，我们可以使用这两个图像来提取它们的信息。
&emsp;当我们说匹配点时，一般来说，我们指的是场景中容易识别的特征。我们称这些特征为特征。
&emsp;那么，一个特征应该具有哪些特征呢？它必须是唯一可识别的。
## 图像特征类型
举几个例子：</br>
  -  边缘</br>
  - 角点（也称为兴趣点）</br>
  - Blobs（也称为感兴趣的区域）</br>
在本教程中，我们将特别研究角点特征。
## 为什么角点这么特别？
&emsp;因为它是两条边的交集,它表示这两条边的方向改变的点。因此，图像的梯度（在两个方向上）有很大的变化，可以用来检测它。
## 它是如何工作的？
&emsp;我们寻找一个角点，由于角点代表图像中渐变的变化，我们将寻找这种“变化”。</br>
考虑一个灰度图像$(I)$。我们将扫描一个窗口$w(x，y)$,(位移u在x方向，v在y方向)$I$并计算强度的变化。<br/>
$$  E(x,y) = \sum_{x,y}w(x,y)[I(x+u,y+v)−I(x,y)]^2$$
其中：<br/>
- $w(x,y)$ 表示在窗口中的位置$(x,y)$。
- $I(x,y)$表示在 $(x,y)$位置的灰度值。
- $I(x+u，y+v)$表示滑动窗口$(x+u，y+v)$处的强度。<br/>
因为我们要找有角点的窗口，所以我们要找强度变化较大的窗口。因此，我们必须最大化上述等式，特别是术语:</br>
$$\sum_{x,y}w(x,y)[I(x+u,y+v)−I(x,y)]^2$$
- 使用泰勒表达式展开：<br>
$$  E(u,v) \approx \sum_{x,y}[I(x,y) + uI_x + vI_y - I(x,y)  ]^2$$
- 展开方程：<br/>
$$  E(u,v) \approx \sum_{x,y}u^2I_x^2 + 2uvI_xI_y + v^2I_y^2 $$
可以用矩阵形式表示为：
$$  E(u,v) \approx  [u ~ v]    \left (   \sum_{x,y} w(x,y)    \left[ \begin{matrix}    I_x^2 & I_xI_y \\  
I_xI_y & I_y^2  \end{matrix}  \right]  \right)  \left[\begin{matrix}  u \\  v \end{matrix} \right]$$
- 我们来表示：
 $$ M = \sum_{x,y}w(x,y)\left[ \begin{matrix}    I_x^2 & I_xI_y \\ 
 I_xI_y & I_y^2  \end{matrix}  \right]$$
 - 所以，我们现在的方程式是：<br/>
  $$  E(u,v)   \approx  [u ~ v]M \left[\begin{matrix}  u \\  v \end{matrix} \right]$$
  - 为每个窗口计算分数，以确定它是否可能包含角点：
  $$R=det(M)−k(trace(M))^2$$
  其中：

  - $$ det(M) =  \lambda _1\lambda _2$$
  - $$trace(M) = \lambda _1 + \lambda _2 $$
  分数$R$大于某个值的窗口被视为“角点”。
  代码清单<br/>
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
void cornerHarris_demo( int, void* );
int main( int argc, char** argv )
{
    CommandLineParser parser( argc, argv, "{@input | building.jpg | input image}" );
    src = imread( samples::findFile( parser.get<String>( "@input" ) ) );
    if ( src.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }
    cvtColor( src, src_gray, COLOR_BGR2GRAY );
    namedWindow( source_window );
    createTrackbar( "Threshold: ", source_window, &thresh, max_thresh, cornerHarris_demo );
    imshow( source_window, src );
    cornerHarris_demo( 0, 0 );
    waitKey();
    return 0;
}
void cornerHarris_demo( int, void* )
{
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    Mat dst = Mat::zeros( src.size(), CV_32FC1 );
    cornerHarris( src_gray, dst, blockSize, apertureSize, k );
    Mat dst_norm, dst_norm_scaled;
    normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs( dst_norm, dst_norm_scaled );
    for( int i = 0; i < dst_norm.rows ; i++ )
    {
        for( int j = 0; j < dst_norm.cols; j++ )
        {
            if( (int) dst_norm.at<float>(i,j) > thresh )
            {
                circle( dst_norm_scaled, Point(j,i), 5,  Scalar(0), 2, 8, 0 );
            }
        }
    }
    namedWindow( corners_window );
    imshow( corners_window, dst_norm_scaled );
}
  ```
<!-- more -->
<!-- more -->
## 解释
### 结果
原始图像：</br>
![](file://C:/Users/yee/Documents/Gridea/post-images/1585153255589.JPG)
检测到的角被一个黑色的小圆圈包围。</br>
![](file://C:/Users/yee/Documents/Gridea/post-images/1585153274034.JPG)
