#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <algorithm>
#include <Eigen/Dense>
using namespace std;


template<typename T>
class my_img
{
  T *F = nullptr;
  int    w;
  int    h;
public:
  my_img(int iw,int ih) : w{iw},h{ih} { cout << "alloc myimg" << endl; F = new T[w*h];}
  T& at(int y,int x) const {return F[w*y+x];}
  ~my_img() { cout << "Free myimg" << endl; delete [] F;}
};

const int QT[8][8] = {{16, 11, 10, 16, 24, 40, 51, 61},
                    {12, 12, 14, 19, 26, 58, 60, 55},
                    {14, 13, 16, 24, 40, 57, 69, 56},
                    {14, 17, 22, 29, 51, 87, 80, 62},
                    {18, 22, 37, 56, 68, 109, 103, 77},
                    {24, 35, 55, 64, 81, 104, 113, 92},
                    {49, 64, 78, 87, 103, 121, 120, 101},
                    {72, 92, 95, 98, 112, 100, 103, 99}};

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

cv::Mat RGB2YCbCr(cv::Mat img){

  int w = img.cols;
  int h = img.rows;
  cv::Mat out = cv::Mat::zeros(h,w,CV_8UC3);

  for(int y = 0;y<h;y++) {
    for(int x  = 0;x<w;x++) {
      // Y
      out.at<cv::Vec3b>(y, x)[0] = (int)((float)img.at<cv::Vec3b>(y,x)[0] * 0.114 + \
				  (float)img.at<cv::Vec3b>(y,x)[1] * 0.5870 + \
				  (float)img.at<cv::Vec3b>(y,x)[2] * 0.299);

      // Cb
      out.at<cv::Vec3b>(y, x)[1] = (int)((float)img.at<cv::Vec3b>(y,x)[0] * 0.5 + \
				  (float)img.at<cv::Vec3b>(y,x)[1] * (-0.3323) + \
				  (float)img.at<cv::Vec3b>(y,x)[2] * (-0.1687) + 128);

      // Cr
      out.at<cv::Vec3b>(y, x)[2] = (int)((float)img.at<cv::Vec3b>(y,x)[0] * (-0.0813) + \
				  (float)img.at<cv::Vec3b>(y,x)[1] * (-0.4187) + \
				  (float)img.at<cv::Vec3b>(y,x)[2] * 0.5 + 128);
    }
  }

  return out;
}

cv::Mat YCbCr2BGR(cv::Mat ycbcr, cv::Mat out){

  int width = out.rows;
  int height = out.cols;
  
  for (int j = 0; j < height; j ++){
    for (int i = 0; i < width; i ++){
      // R
      out.at<cv::Vec3b>(j, i)[2] = (uchar)(ycbcr.at<cv::Vec3b>(j, i)[0] + (ycbcr.at<cv::Vec3b>(j, i)[2] - 128) * 1.4102);

      // G
      out.at<cv::Vec3b>(j, i)[1] = (uchar)(ycbcr.at<cv::Vec3b>(j, i)[0] - (ycbcr.at<cv::Vec3b>(j, i)[1] - 128) * 0.3441 - (ycbcr.at<cv::Vec3b>(j, i)[2] - 128) * 0.7139);

      // B
      out.at<cv::Vec3b>(j, i)[0] = (uchar)(ycbcr.at<cv::Vec3b>(j, i)[0] + (ycbcr.at<cv::Vec3b>(j, i)[1] - 128) * 1.7718);
    }
  }
  return out;
}


double getC(int i){
  if(i==0) {
    return 1/sqrt(2);
  }else{
    return 1;
  }
}

template <typename Ty>
my_img<Ty>& dct(cv::Mat img, my_img<Ty>& dct_store,int grid,const int qt[8][8] = nullptr)
{
  int w = img.cols;
  int h = img.rows;
  double I = 0;
  double val;

  for(int vv=0;vv<h/grid;vv++) {
    for(int uu=0;uu<w/grid;uu++) {
      for(int v=0;v<grid;v++) {
        for(int u=0;u<grid;u++) {
          I = 0;
          for(int y = 0 ; y < grid; y++ ) {
            for(int x = 0; x < grid ; x++ ) {
              val   = (double)img.at<uchar>(vv*grid+y,uu*grid+x);
              I += val * cos((2*x+1)*u*M_PI/(2*grid))*cos((2*y+1)*v*M_PI/(2*grid));
            }
          }
          val = 2*getC(u)*getC(v)*I/grid;
          if(qt) {
            val= roundf(val/qt[v][u]);
          }
          dct_store.at(vv*grid+v,uu*grid+u)=val;
        }
      }
    }
  }

  return dct_store;
}

template <typename Ty>
cv::Mat idct(my_img<Ty>& dct_store,cv::Mat img,int grid,int K,const int qt[8][8] = nullptr)
{
  int w = img.cols;
  int h = img.rows;
  double F;
  double val;

  for(int yy=0;yy<h/grid;yy++) {
    for(int xx=0;xx<w/grid;xx++) {
      for(int y=0;y<grid;y++) {
        for(int x=0;x<grid;x++) {
          F = 0;
          for(int v = 0; v < K; v++ ) {
            for(int u = 0; u < K ; u++ ) {
              val   = dct_store.at(yy*grid+v,xx*grid+u)*getC(u)*getC(v);
              if(qt) {
                val=roundf(val*qt[v][u]);
              }
              F += val * cos((2*x+1)*u*M_PI/(2*grid))*cos((2*y+1)*v*M_PI/(2*grid));
            }
          }
          val = 2.0*F/grid;
          val = fmin(val,255);
          val = fmax(val,0);
          img.at<uchar>(yy*grid+y,xx*grid+x) = val;
        }
      }
    }
  }

  return img;
}

cv::Mat answer36(cv::Mat img)
{
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  int ow = w;
  int oh = h;
  int grid = 8;
  my_img<double> dct_store{w,h};
  cv::Mat out = cv::Mat::zeros(h,w,CV_8UC1);
  cv::Mat imgY = RGB2Y(img);

  dct_store = dct<double>(imgY,dct_store,grid);
  out = idct<double>(dct_store,out,grid,grid);

  return out;
}

double calc_psnr(cv::Mat img1,cv::Mat img2)
{
  int w = img1.cols;
  int h = img1.rows;  
  double mse=0;

  for(int y=0;y<h;y++) {
    for(int x=0;x<w;x++){
      double val = img1.at<uchar>(y,x) - img2.at<uchar>(y,x);
      val*=val;
      mse+= val;
    }
  }
  mse /= (h*w);

  return 10*log10(255*255/mse);
}

double calc_bitrate(double K){
  return 8*K*K/(8*8);
}

cv::Mat answer37(cv::Mat img)
{
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  int ow = w;
  int oh = h;
  int grid = 8;
  int K = 7;
  my_img<double> dct_store{w,h};
  cv::Mat out = cv::Mat::zeros(h,w,CV_8UC1);
  cv::Mat imgY = RGB2Y(img);

  dct_store = dct<double>(imgY,dct_store,grid);
  out = idct<double>(dct_store,out,grid,K);

  cout << "PSRN:" << calc_psnr(imgY,out) << endl;
  cout << "Bitrate:" << calc_bitrate(K) << endl;

  return out;
}




cv::Mat answer38(cv::Mat img)
{
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  int ow = w;
  int oh = h;
  int grid = 8;
  int K = 4;
  my_img<double> dct_store{w,h};
  cv::Mat out = cv::Mat::zeros(h,w,CV_8UC1);
  cv::Mat imgY = RGB2Y(img);

  cv::imwrite("out_org.jpg",imgY);

  dct_store = dct<double>(imgY,dct_store,grid,QT);
  out = idct<double>(dct_store,out,grid,K,QT);

  cout << "PSRN:" << calc_psnr(imgY,out) << endl;
  cout << "Bitrate:" << calc_bitrate(K) << endl;

  cv::imwrite("out.jpg",out);

  return out;
}


cv::Mat answer39(cv::Mat img)
{
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  int ow = w;
  int oh = h;
  int grid = 8;
  int K = 4;
  my_img<double> dct_store{w,h};
  cv::Mat out = cv::Mat::zeros(h,w,CV_8UC3);
  cv::Mat imgCbCr = RGB2YCbCr(img);

  for(int y = 0;y<h;y++) {
    for(int x  = 0;x<w;x++) {
      imgCbCr.at<cv::Vec3b>(y,x)[0] *= 0.7;
    }
  }

  out = YCbCr2BGR(imgCbCr,out);


  // dct_store = dct<double>(imgY,dct_store,grid,QT);
  // out = idct<double>(dct_store,out,grid,K,QT);
  // cout << "PSRN:" << calc_psnr(imgY,out) << endl;
  // cout << "Bitrate:" << calc_bitrate(K) << endl;

  return out;
}

int main(int argc, const char* argv[]){
  cv::Mat img = cv::imread("../Question_31_40/imori.jpg",cv::IMREAD_COLOR);
  cv::Mat out = answer39(img);
  
  cv::imshow("sample", out);
  cv::waitKey(0);
  cv::destroyAllWindows(); 

  return 0;
}