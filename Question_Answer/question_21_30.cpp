#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <algorithm>

using namespace std;

cv::Mat HistNorm(cv::Mat img) {
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();

  cv::Mat out = cv::Mat::zeros(w,h,CV_8UC3);

  uchar _max=0;
  uchar _min= 255;
  uchar   color;
  double  c;

  for(int ch=0;ch<ch_num;ch++) {
    for(int j = 0;j<h;j++) {
      for(int i  = 0;i<w;i++) {
        color = img.at<cv::Vec3b>(j,i)[ch];
        _max = max(color,_max);
        _min = min(color,_min);
      }
    }
  }
  
  for(int ch=0;ch<ch_num;ch++) {
    for(int j = 0;j<h;j++) {
      for(int i  = 0;i<w;i++) {
        c = (double)(255 - 0)/(_max-_min) * (img.at<cv::Vec3b>(j,i)[ch] - _min);
        out.at<cv::Vec3b>(j,i)[ch] = (uchar)c;
      }
    }
  }

  return out;
}


cv::Mat answer20(cv::Mat img)
{
  return HistNorm(img);
}

cv::Mat HistNorm2(cv::Mat img,int m0=128,int s0=52) {
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();

  cv::Mat out = cv::Mat::zeros(w,h,CV_8UC3);

  double  c,s,sum;
  uchar   ave,color;

  sum=0;
  for(int ch=0;ch<ch_num;ch++) {
    for(int j = 0;j<h;j++) {
      for(int i  = 0;i<w;i++) {
        sum += img.at<cv::Vec3b>(j,i)[ch];
      }
    }
  }

  ave = sum/(w*h*ch_num);

  sum=0;
  for(int ch=0;ch<ch_num;ch++) {
    for(int j = 0;j<h;j++) {
      for(int i  = 0;i<w;i++) {
        sum += (ave-img.at<cv::Vec3b>(j,i)[ch])*(ave-img.at<cv::Vec3b>(j,i)[ch]);
      }
    }
  }

  s = sqrt(sum/(w*h*ch_num));

  cout << (int)ave << "," << s << endl;
  
  for(int ch=0;ch<ch_num;ch++) {
    for(int j = 0;j<h;j++) {
      for(int i  = 0;i<w;i++) {
        c =  (double)s0/s * (img.at<cv::Vec3b>(j,i)[ch] - ave) + m0;
        out.at<cv::Vec3b>(j,i)[ch] = (uchar)c;
      }
    }
  }

  return out;
}

cv::Mat answer21(cv::Mat img)
{
  return HistNorm2(img);
}


cv::Mat HistNorm3(cv::Mat img,int m0=128,int s0=52) {
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();

  cv::Mat out = cv::Mat::zeros(w,h,CV_8UC3);
  double Zmax = 255;
  double hist[255]={};
  double S = h*w*ch_num;
  int val;
  double hist_sum = 0;

  for(int ch=0;ch<ch_num;ch++) {
    for(int j = 0;j<h;j++) {
      for(int i  = 0;i<w;i++) {
        val = img.at<cv::Vec3b>(j,i)[ch];
        hist[val]++;
      }
    }
  }

  for(int ch=0;ch<ch_num;ch++) {
    for(int j = 0;j<h;j++) {
      for(int i  = 0;i<w;i++) {
        val = img.at<cv::Vec3b>(j,i)[ch];

        hist_sum = 0;
        for(int l=0;l<val;l++){
          hist_sum += hist[l];
        }

        out.at<cv::Vec3b>(j,i)[ch] = (uchar)(Zmax / S * hist_sum);
      }
    }
  }

  return out;
}

cv::Mat answer22(cv::Mat img)
{
  return HistNorm3(img);
}


cv::Mat gamma_correct(cv::Mat img,int c,double gamma=52) {
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();

  cv::Mat out = cv::Mat::zeros(w,h,CV_8UC3);

  int val;
  double val2 = 0;

  for(int ch=0;ch<ch_num;ch++) {
    for(int j = 0;j<h;j++) {
      for(int i  = 0;i<w;i++) {
        val  = img.at<cv::Vec3b>(j,i)[ch];
        val2 = pow((double)val/255,1/gamma)*255;
        out.at<cv::Vec3b>(j,i)[ch] = (uchar)val2;
      }
    }
  }

  return out;
}

cv::Mat answer23(cv::Mat img)
{
  return gamma_correct(img,1,2.2);
}


cv::Mat NearestNeighbor(cv::Mat img,int wa,int ha) {
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();

  cv::Mat out = cv::Mat::zeros(ha*h,wa*w,CV_8UC3);

  int val;
  double val2 = 0;

  for(int ch=0;ch<ch_num;ch++) {
    for(int j = 0;j<h;j++) {
      for(int i  = 0;i<w;i++) {
        val  = img.at<cv::Vec3b>(j,i)[ch];
        for(int jj = 0;jj<ha+1;jj++) {
          for(int ii = 0;ii<wa+1;ii++) {
            out.at<cv::Vec3b>(j*ha+jj,i*wa+ii)[ch] = val;
          }
        }
      }
    }
  }

  return out;
}

cv::Mat answer24(cv::Mat img)
{
  return NearestNeighbor(img,4,2);
}

cv::Mat Bilinear(cv::Mat img,double mag = 1.5) {
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  int ow = mag*w;
  int oh = mag*h;

  cv::Mat out = cv::Mat::zeros(oh,ow,CV_8UC3);

  double val = 0;

  for(int ch=0;ch<ch_num;ch++) {
    for(int j = 0;j<oh;j++) {
      for(int i  = 0;i<ow;i++) {
        val = 0;

        int ii = floor((double)i/mag);
        int jj = floor((double)j/mag);
        double ii_a = (double)i/mag - ii;
        double jj_b = (double)j/mag - jj;

        // cout << ii_a << "<" << jj_b << endl;

        val =  (1.0-jj_b)*(1.0-ii_a)* img.at<cv::Vec3b>(jj  ,ii)[ch] + (1.0-jj_b)*(ii_a)*img.at<cv::Vec3b>(jj  ,ii+1)[ch] + \
               (jj_b)*(1.0-ii_a)    * img.at<cv::Vec3b>(jj+1,ii)[ch] + (jj_b)*(ii_a)*img.at<cv::Vec3b>(jj+1,ii+1)[ch];

        out.at<cv::Vec3b>(j,i)[ch] = (uchar)val;
      }
    }
  }

  return out;
}

cv::Mat answer25(cv::Mat img)
{
  return Bilinear(img,1.5);
}

// weight function
double weight(double t){
  double a = -1;
  if (fabs(t) <= 1){
    return (a + 2) * pow(fabs(t), 3) - (a + 3) * pow(fabs(t), 2) + 1;
  } else if(fabs(t) <= 2){
    return a * pow(fabs(t), 3) - 5 * a * pow(fabs(t), 2) + 8 * a * fabs(t) - 4 * a;
  } 
  return 0;
}

// clip value [*, *] -> [min, max]
int val_clip(int x, int min, int max){
  return fmin(fmax(x, min), max);
}

cv::Mat Bicubic(cv::Mat img,double mag = 1.5) {
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  int ow = mag*w;
  int oh = mag*h;

  cv::Mat out = cv::Mat::zeros(oh,ow,CV_8UC3);
  int x_before, y_before;
  double val = 0;
  double dx, dy, wx, wy, w_sum;
  int _x, _y;

  for(int ch=0;ch<ch_num;ch++) {
    for(int j = 0;j<oh;j++) {
      dy = j / mag;
      y_before = (int)floor(dy);
      
      for(int i  = 0;i<ow;i++) {
        w_sum = 0;
        val   = 0;

        dx = i / mag;
        x_before = (int)floor(dx);
        
        for (int jj = -1; jj < 3; jj++){
          _y = val_clip(y_before + jj, 0, h - 1);
          wy = weight(fabs(dy - _y));
          for (int ii = -1; ii < 3; ii++){
            _x = val_clip(x_before + ii, 0, w - 1);
            wx = weight(fabs(dx - _x));
            w_sum += wy*wx;
            val += (double)img.at<cv::Vec3b>(_y,_x)[ch]* wx * wy;
          }
        }
        val /= w_sum;
        val = val_clip(val,0,255);
        out.at<cv::Vec3b>(j,i)[ch] = (uchar)val;
      }
    }
  }

  return out;
}


cv::Mat answer26(cv::Mat img)
{
  return Bicubic(img,1.5);
}

#include <opencv2/imgproc.hpp>  // warpAffine を使うために必要
#include <opencv2/opencv.hpp>

cv::Mat affine(cv::Mat img, double a, double b, double c, double d, double tx, double ty)
{
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  int ow = a*w;
  int oh = d*h;

  cv::Mat out = cv::Mat::zeros(oh,ow,CV_8UC3);

  double a0 = (a*d - b *c);

  for(int ch=0;ch<ch_num;ch++) {
    for(int y=0;y<oh;y++) {
      for(int x=0;x<ow;x++) {
          int xx = ( d*x - b*y)/a0 - tx;
          int yy = (-c*x + a*y)/a0 - ty;
          int value= 0;
          // xx = min(max(xx,0 ),255);
          // yy = min(max(xx,0 ),255);
          if( xx >=0 and xx <w and yy >= 0 and yy < h) {
            value = img.at<cv::Vec3b>(yy,xx)[ch];
          }else{
            value = 0;
          }
        out.at<cv::Vec3b>(y,x)[ch] = value;
      } 
    }
  }

  return out;
}

cv::Mat answer28(cv::Mat img)
{
  return affine(img,1,0,0,1,30,-30);
}

cv::Mat answer29(cv::Mat img)
{
  return affine(img,1.3,0,0,0.8,0,0);
}

cv::Mat answer30(cv::Mat img)
{
  double angle = -30 * M_PI / 180;

  //center
  double cx = img.cols/2.;
  double cy = img.rows/2.;
  double a0 = (cos(angle)*cos(angle)+ sin(angle) *sin(angle));
  double n_cx = ( cos(angle)*cx + sin(angle)*cy) / a0;
  double n_cy = (-sin(angle)*cx + cos(angle)*cy) / a0;
  double tx = n_cx - cx;
  double ty = n_cy - cy;

  return affine(img,cos(angle),-sin(angle),sin(angle),cos(angle),tx,ty);
}

int main(int argc, const char* argv[]){
  cv::Mat img = cv::imread("../Question_21_30/imori.jpg",cv::IMREAD_COLOR);
  cv::Mat out = answer30(img);
  cv::imshow("sample", out);
  cv::waitKey(0);
  cv::destroyAllWindows(); 

  return 0;
}