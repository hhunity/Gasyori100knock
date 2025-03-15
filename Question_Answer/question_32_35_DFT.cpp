#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <algorithm>
#include <Eigen/Dense>
using namespace std;

const int height = 128 , width = 128;

struct fourier_str{
  complex<double> coef[height][width];
};

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


fourier_str dft(cv::Mat img, fourier_str fs)
{
  double I = 0;
  double theta;
  complex<double> val;

  for(int l=0;l<height;l++) {
    for(int k=0;k<width;k++) {
      val.real(0);
      val.imag(0);
      for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
        // I     = img.at<uchar>(y,x);
        I     = img.at<uchar>(y,x) * powf(-1.0,(float)(x+y));
        theta = -2*M_PI*((double)k * (double)x/(double)width+(double)l*(double)y / (double)height);
        val   += complex<double>(cos(theta),sin(theta)) * I;
        }
      }
      val /= sqrt(height*width);
      // cout << val << endl;
      fs.coef[l][k]=val;
    }
  }

  return fs;
}

fourier_str filter_cut(fourier_str fs,bool (*comp)(double,double))
{
  double r;
  double hw=width/2,hh=height/2;
  for(int l=0;l<height;l++) {
    for(int k=0;k<width;k++) {

      r = ((double)k-hw)*((double)k-hw)+((double)l-hh)*((double)l-hh);
      r = sqrt(r);
      // cout << l <<"," << k << ":" << r << endl;
      if(comp(r,hh)) {
        fs.coef[l][k] = 0;
      }
    }
  }
  return fs;
}

cv::Mat idft(fourier_str fs,cv::Mat img)
{
  double theta;
  double g;
  complex<double> G;
  complex<double> val;

  for (int y = 0; y < height; y++){
    for (int x = 0; x < width; x++){
      val.real(0);
      val.imag(0);
      for(int l=0;l<height;l++) {
        for(int k=0;k<width;k++) {;
        G     = fs.coef[l][k];
        theta = 2*M_PI*((double)k * (double)x/(double)width+(double)l*(double)y / (double)height);

        val   += complex<double>(cos(theta),sin(theta)) * G;
        }
      }
      //絶対値なので
      val=val*(double)powf(-1.0,(float)(x+y));
      g = abs(val)/sqrt(height*width);
      // img.at<uchar>(y,x)=(uchar)g;
      img.at<uchar>(y,x)=(uchar)g;
    }
  }

  return img;
}

cv::Mat answer32(cv::Mat img)
{
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  int ow = w;
  int oh = h;
  fourier_str fr;
  cv::Mat out = cv::Mat::zeros(h,w,CV_8UC1);
  cv::Mat imgY = RGB2Y(img);

  fr  = dft(imgY,fr);
  fr  = filter_cut(fr,[](double r,double a){return r>=a*0.5; });
  out = idft(fr,out);

  return out;
}

cv::Mat answer33(cv::Mat img)
{
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  int ow = w;
  int oh = h;
  fourier_str fr;
  cv::Mat out = cv::Mat::zeros(h,w,CV_8UC1);
  cv::Mat imgY = RGB2Y(img);

  fr  = dft(imgY,fr);
  fr  = filter_cut(fr,[](double r,double a){return r<=a*0.1; });
  out = idft(fr,out);

  return out;
}

cv::Mat answer34(cv::Mat img)
{
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  int ow = w;
  int oh = h;
  fourier_str fr;
  cv::Mat out = cv::Mat::zeros(h,w,CV_8UC1);
  cv::Mat imgY = RGB2Y(img);

  fr  = dft(imgY,fr);
  fr  = filter_cut(fr,[](double r,double a){return (r>=a*0.1) and (r<=a*0.5); });
  out = idft(fr,out);

  return out;
}

int main(int argc, const char* argv[]){
  cv::Mat img = cv::imread("../Question_31_40/imori.jpg",cv::IMREAD_COLOR);
  cv::Mat out = answer33(img);
  cv::imshow("sample", out);
  cv::waitKey(0);
  cv::destroyAllWindows(); 

  return 0;
}