#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <algorithm>
#include <Eigen/Dense>
using namespace std;

cv::Mat RGB2Y(cv::Mat img){

  int w = img.cols;
  int h = img.rows;
  cv::Mat out = cv::Mat::zeros(h,w,CV_8UC1);

  for(int y = 0;y<h;y++) {
    for(int x  = 0;x<w;x++) {
      out.at<uchar>(y,x) = 0.2126 * (float)img.at<cv::Vec3b>(y,x)[2] + \
                                   0.7152 * (float)img.at<cv::Vec3b>(y,x)[1] + \
                                   0.0722 * (float)img.at<cv::Vec3b>(y,x)[0];
    }
  }

  return out;
}

cv::Mat Y_Th2(cv::Mat img, int th){

  int w = img.cols;
  int h = img.rows;
  uchar gray;

  for(int y = 0;y<h;y++) {
    for(int x  = 0;x<w;x++) {
      gray = img.at<uchar>(y,x);
      img.at<uchar>(y,x) = gray < th ? 0 : 255;
    
    }
  }

  return img;
}

int calc_Sb2(cv::Mat img,int t,float &W0,float &W1,float &M0,float &M1)
{
    int c0,c1;
    int sum0,sum1,Sb2;
    uchar gray;
    c0=c1=0;
    sum0=sum1=0;
    int w = img.rows;
    int h = img.cols;
    for(int y = 0;y < h;y++) {
      for(int x = 0;x < w;x++) {
        gray = img.at<uchar>(y,x);
        if( gray < t) {
            c0++;
            sum0+=gray;
        }else{
            c1++;
            sum1+=gray;
        }
      }
    }

    W0 = (float)c0/(c0+c1);
    W1 = (float)c1/(c0+c1);
    M0 = (float)sum0/c0;
    M1 = (float)sum1/c1;

    Sb2 = W0*W1*(M0 - M1)*(M0 - M1);

    // cout << "Sb2 :" << Sb2 << "," << W0 << "," << W1 <<"," << W0 + W1 << "," << M0 <<"," << M1  << endl;

    return Sb2;
}

cv::Mat OOTH(cv::Mat img)
{
  float   W0,W1;
  float   M0,M1;
  int     Sb2,maxSb2 = 0,th = 0;

  for(int i =0 ;i<255;i++) {
    Sb2 = calc_Sb2(img,i,W0,W1,M0,M1);
    if(Sb2 > maxSb2) {
      maxSb2 = Sb2;
      th = i;
    }
  }

  cout << "maxSb2 :" << maxSb2 << " th =" << th << endl;

  return Y_Th2(img,th);
}

cv::Mat MorphologicalProcess(cv::Mat img,bool (*comp)(int))
{
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  cv::Mat out = img.clone();
  int fil[3][3] ={{0,1,0},{1,0,1},{0,1,0}};
  int val;

  for(int y = 0;y < h;y++) {
    for(int x = 0;x < w;x++) {
      val = 0;
      // if(img.at<uchar>(y,x)!=255) {
      //   out.at<uchar>(y,x) = 0;
      //   continue;
      // };
      for(int yy=0;yy<3;yy++) {
        for(int xx=0;xx<3;xx++) {
          int iy = fmin(fmax(y+(yy-1),0),h-1);
          int ix = fmin(fmax(x+(xx-1),0),w-1);
          // cout << iy << "," << ix << endl;
          val += img.at<uchar>(iy,ix)*fil[yy][xx];
        }
      }
      // cout << val << "," << 255*4 << endl;
      if(comp(val)) {
        out.at<uchar>(y,x) = 0;
      }
    }
  }
  
  return out;
}

cv::Mat MorphologicalDilation(cv::Mat img)
{
  return MorphologicalProcess(img ,[](int a){return a < 255;});
}
cv::Mat MorphologicalErogion(cv::Mat img)
{
  return MorphologicalProcess(img ,[](int a){return a < 255*4;});
}
cv::Mat answer47(cv::Mat img)
{
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  int ow = w;
  int oh = h;

  cv::Mat imgY    = RGB2Y(img);
  cv::Mat imgY_th = OOTH(imgY);
  cv::Mat imgY_mo1= MorphologicalDilation(imgY_th);
  cv::Mat imgY_mo2= MorphologicalDilation(imgY_mo1);

  return imgY_mo2;
}

cv::Mat answer48(cv::Mat img)
{
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  int ow = w;
  int oh = h;

  cv::Mat imgY    = RGB2Y(img);
  cv::Mat imgY_th = OOTH(imgY);
  cv::Mat imgY_mo1= MorphologicalErogion(imgY_th);
  cv::Mat imgY_mo2= MorphologicalErogion(imgY_mo1);

  return imgY_mo2;
}

cv::Mat answer49(cv::Mat img)
{
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  int ow = w;
  int oh = h;

  cv::Mat imgY    = RGB2Y(img);
  cv::Mat imgY_th = OOTH(imgY);
  cv::Mat imgY_mo1= MorphologicalDilation(imgY_th);
  cv::Mat imgY_mo2= MorphologicalErogion(imgY_mo1);

  return imgY_mo2;
}

int main(int argc, const char* argv[]){
  // read image
  cv::Mat img = cv::imread("../Question_41_50/imori.jpg", cv::IMREAD_COLOR);

  // Canny
  cv::Mat edge = answer49(img);

  //cv::imwrite("out.jpg", out);
  cv::imshow("answer(edge)", edge);
  cv::waitKey(0);
  cv::destroyAllWindows();

  return 0;
}