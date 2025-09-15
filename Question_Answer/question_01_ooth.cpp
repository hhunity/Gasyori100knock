#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <algorithm>

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

int calc_Sb2(cv::Mat img,int t,float &W0,float &W1,float &M0,float &M1)
{
    int c0,c1;
    int sum0,sum1,Sb2;
    uchar gray;
    c0=c1=0;
    sum0=sum1=0;
    int w = img.rows;
    int h = img.cols;

    int del = 4;

    for(int y = 0;y < h;y=y+del) {
      for(int x = 0;x < w;x=x+del) {
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

    //cout << "Sb2 :" << Sb2 << "," << W0 << "," << W1 <<"," << W0 + W1 << "," << M0 <<"," << M1  << endl;

    return Sb2;
}

cv::Mat OOTH(cv::Mat imgY){

  float   W0,W1;
  float   M0,M1;
  int     Sb2,maxSb2 = 0,th = 0;
  int w = imgY.rows;
  int h = imgY.cols;
  
  for(int i =0 ;i<255;i++) {
    Sb2 = calc_Sb2(imgY,i,W0,W1,M0,M1);
    if(Sb2 > maxSb2) {
      maxSb2 = Sb2;
      th = i;
    }
  }

  // prepare output
  cv::Mat out = cv::Mat::zeros(h, w, CV_8UC1);

  // each y, x
  for (int y = 0; y < h; y++){
    for (int x = 0; x < w; x++){
      // Binarize
      if (imgY.at<uchar>(y, x) > th){
        out.at<uchar>(y, x) = 255;
      } else {
        out.at<uchar>(y, x) = 0;
      }
    
    }
  }


  //cout << "maxSb2 :" << maxSb2 << " th =" << th << endl;

  return out;
}



int main(int argc, const char* argv[]){
  cv::Mat img  = cv::imread("../tutorial/imori.jpg",cv::IMREAD_COLOR);
  cv::Mat imgY = RGB2Y(img);


	auto start = std::chrono::high_resolution_clock::now();
  cv::Mat out;
#if 1
  out = OOTH(imgY);
#else
  cv::threshold(imgY, out, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
#endif

  auto end   = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  std::cout << "time:" << duration.count()*4/1000 <<  std::endl;


  // cv::imshow("sample", out);
  // cv::waitKey(0);
  // cv::destroyAllWindows(); 

  return 0;
}