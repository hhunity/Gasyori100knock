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
  int    ch;
public:
  my_img(int iw,int ih) : w{iw},h{ih},ch{1} { cout << "alloc myimg" << endl; F = new T[w*h];}
  my_img(int iw,int ih,int ich) : w{iw},h{ih},ch{ich} { cout << "alloc myimg" << endl; F = new T[ich*w*h];}
  T& at(int y,int x) const {return F[w*y+x];}
  T& at(int y,int x,int c) const {return F[c*(w*h)+(w*y)+x];}
  ~my_img() { cout << "Free myimg" << endl; delete [] F;}
};

const int QTY[8][8] = {{16, 11, 10, 16, 24, 40, 51, 61},
                    {12, 12, 14, 19, 26, 58, 60, 55},
                    {14, 13, 16, 24, 40, 57, 69, 56},
                    {14, 17, 22, 29, 51, 87, 80, 62},
                    {18, 22, 37, 56, 68, 109, 103, 77},
                    {24, 35, 55, 64, 81, 104, 113, 92},
                    {49, 64, 78, 87, 103, 121, 120, 101},
                    {72, 92, 95, 98, 112, 100, 103, 99}};

const int QTCbCr[8][8] = {{17, 18, 24, 47, 99, 99, 99, 99},
                    {18, 21, 26, 66, 99, 99, 99, 99},
                    {24, 26, 56, 99, 99, 99, 99, 99},
                    {47, 66, 99, 99, 99, 99, 99, 99},
                    {99, 99, 99, 99, 99, 99, 99, 99},
                    {99, 99, 99, 99, 99, 99, 99, 99},
                    {99, 99, 99, 99, 99, 99, 99, 99},
                    {99, 99, 99, 99, 99, 99, 99, 99}};

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
  int Y,Cb,Cr;
  cv::Mat out = cv::Mat::zeros(h,w,CV_8UC3);

  for(int y = 0;y<h;y++) {
    for(int x  = 0;x<w;x++) {
      // Y
      Y = (int)((float)img.at<cv::Vec3b>(y,x)[0] * 0.114 + \
				  (float)img.at<cv::Vec3b>(y,x)[1] * 0.5870 + \
				  (float)img.at<cv::Vec3b>(y,x)[2] * 0.299);

      // Cb
      Cb = (int)((float)img.at<cv::Vec3b>(y,x)[0] * 0.5 + \
				  (float)img.at<cv::Vec3b>(y,x)[1] * (-0.3323) + \
				  (float)img.at<cv::Vec3b>(y,x)[2] * (-0.1687) + 128);

      // Cr
      Cr = (int)((float)img.at<cv::Vec3b>(y,x)[0] * (-0.0813) + \
				  (float)img.at<cv::Vec3b>(y,x)[1] * (-0.4187) + \
				  (float)img.at<cv::Vec3b>(y,x)[2] * 0.5 + 128);

      Y=fmin(255,fmax(0,Y));
      Cb=fmin(255,fmax(0,Cb));
      Cr=fmin(255,fmax(0,Cr));
      out.at<cv::Vec3b>(y, x)[0]  = Y;
      out.at<cv::Vec3b>(y, x)[1]  = Cb; 
      out.at<cv::Vec3b>(y, x)[2]  = Cr; 
    }
  }

  return out;
}

cv::Mat YCbCr2BGR(cv::Mat ycbcr, cv::Mat out){

  int width = out.rows;
  int height = out.cols;
  int R=0,G=0,B=0;
  
  for (int j = 0; j < height; j ++){
    for (int i = 0; i < width; i ++){
      // R
      R = (uchar)(ycbcr.at<cv::Vec3b>(j, i)[0] + ((double)ycbcr.at<cv::Vec3b>(j, i)[2] - 128) * 1.4102);
      // G
      G = (uchar)(ycbcr.at<cv::Vec3b>(j, i)[0] - ((double)ycbcr.at<cv::Vec3b>(j, i)[1] - 128) * 0.3441 - ((double)ycbcr.at<cv::Vec3b>(j, i)[2] - 128) * 0.7139);
      // B
      B = (uchar)(ycbcr.at<cv::Vec3b>(j, i)[0] + ((double)ycbcr.at<cv::Vec3b>(j, i)[1] - 128) * 1.7718);

      R=fmin(255,fmax(0,R));
      G=fmin(255,fmax(0,G));
      B=fmin(255,fmax(0,B));

      out.at<cv::Vec3b>(j, i)[2] = (uchar)R;
      out.at<cv::Vec3b>(j, i)[1] = (uchar)G;
      out.at<cv::Vec3b>(j, i)[0] = (uchar)B;
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
my_img<Ty>& dct(const cv::Mat img, my_img<Ty>& dct_store,int grid,const int qtY[8][8],const int qtCbCr[8][8])
{
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  double I = 0;
  double val;
  const int (*qt)[8] = nullptr; // Pointer to a 2D array

  for(int ch=0;ch<ch_num;ch++) {
    qt = (ch==0) ? qtY : qtCbCr;
    for(int vv=0;vv<h/grid;vv++) {
      for(int uu=0;uu<w/grid;uu++) {
        for(int v=0;v<grid;v++) {
          for(int u=0;u<grid;u++) {
            I = 0;
            for(int y = 0 ; y < grid; y++ ) {
              for(int x = 0; x < grid ; x++ ) {
                val   = (double)img.at<cv::Vec3b>(vv*grid+y,uu*grid+x)[ch];
                I += val * cos((2*x+1)*u*M_PI/(2*grid))*cos((2*y+1)*v*M_PI/(2*grid))/grid;
              }
            }
            val = 2*getC(u)*getC(v)*I;
            val= roundf(val/qt[v][u]);
            dct_store.at(vv*grid+v,uu*grid+u,ch)=val;
          }
        }
      }
    }
  }

  return dct_store;
}

template <typename Ty>
cv::Mat idct(my_img<Ty>& dct_store,cv::Mat img,int grid,int K,const int qtY[8][8],const int qtCbCr[8][8])
{
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  double F;
  double val;
  const int (*qt)[8] = nullptr; // Pointer to a 2D array

  for(int ch=0;ch<ch_num;ch++) {
    qt = (ch==0) ? qtY : qtCbCr;
    for(int yy=0;yy<h/grid;yy++) {
      for(int xx=0;xx<w/grid;xx++) {
        for(int y=0;y<grid;y++) {
          for(int x=0;x<grid;x++) {
            F = 0;
            for(int v = 0; v < K; v++ ) {
              for(int u = 0; u < K ; u++ ) {
                val   = dct_store.at(yy*grid+v,xx*grid+u,ch)*getC(u)*getC(v);
                val=roundf(val*qt[v][u]);
                F += val * cos((2*x+1)*u*M_PI/(2*grid))*cos((2*y+1)*v*M_PI/(2*grid))/grid;
              }
            }
            img.at<cv::Vec3b>(yy*grid+y,xx*grid+x)[ch] = (uchar)fmin(fmax(2.0*F, 0), 255);
          }
        }
      }
    }
  }

  return img;
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

cv::Mat answer40(cv::Mat img)
{
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  int ow = w;
  int oh = h;
  int grid = 8;
  int K = 4;
  my_img<double> dct_store{w,h,ch_num};
  cv::Mat out1 = cv::Mat::zeros(h,w,CV_8UC3);
  cv::Mat out2 = cv::Mat::zeros(h,w,CV_8UC3);
  cv::Mat imgCbCr = RGB2YCbCr(img);

  cv::imwrite("out_org.jpg",img);
  // out = YCbCr2BGR(imgCbCr,out);

  dct_store = dct<double>(imgCbCr,dct_store,grid,QTY,QTCbCr);
  out1      = idct<double>(dct_store,out1,grid,K,QTY,QTCbCr);
  out2      = YCbCr2BGR(out1,out2);

  cv::imwrite("out.jpg",out2);

  return out2;
}

int main(int argc, const char* argv[]){
  cv::Mat img = cv::imread("../Question_31_40/imori.jpg",cv::IMREAD_COLOR);
  cv::Mat out = answer40(img);
  
  cv::imshow("sample", out);
  cv::waitKey(0);
  cv::destroyAllWindows(); 

  return 0;
}