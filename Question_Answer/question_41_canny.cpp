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

class canny
{
public:
  cv::Mat edge;
  cv::Mat angle;
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


cv::Mat Gusian(cv::Mat img,double sigma=1.3,int fil_size=3) {
  int w = img.cols;
  int h = img.rows;
  int r = 0;
  int ch_num = img.channels();
  int pad = floor(fil_size/2);
  double filter[fil_size][fil_size];
  double fil_sum =0;

  cv::Mat out = cv::Mat::zeros(h,w,CV_8UC3);

  for(int i =0;i< fil_size;i++) {
    for(int j =0;j<fil_size;j++) {
      int x = j - pad;
      int y = i - pad;
      filter[i][j]=1/(2*M_PI* sigma)*exp(-(x*x+y*y)/(2*sigma*sigma));
      fil_sum+=filter[i][j];
    }
  }

  for(int i =0;i< fil_size;i++) {
    for(int j =0;j<fil_size;j++) {
      filter[i][j]/=fil_sum;
    }
  }

  for(int ch = 0;ch<ch_num;ch++) {
    for(int j = 0;j<h;j++) {
      for(int i  = 0;i<w;i++) {
        r = 0;
        for(int jj = -1;jj <= 1;jj++) {
          for(int ii  = -1;ii <= 1;ii++) {
            if( i+ii>=0 and j+jj>=0 and i+ii<w or j+jj<h) {
              r += filter[ii+1][jj+1] * img.at<cv::Vec3b>(i+ii,j+jj)[ch];
            }
          }
        }
        out.at<cv::Vec3b>(i,j)[ch] = r;
      }
    }
  }
  return out;
}


cv::Mat sobal_filter(cv::Mat img,bool bvertical = false) {
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  double filter[3][3]={{1,2,1},{0,0,0},{-1,-2,-1}};

  if (!bvertical){
    filter[0][1] = 0;
    filter[2][1] = 0;
    filter[1][0] = 2;
    filter[1][2] = -2;
    filter[0][2] = -1;
    filter[2][0] =  1;
    filter[2][2] = -1;
  }

  // for(int j = 0;j<3;j++) {
  //   cout << filter[j][0] << "," << filter[j][1] << "," << filter[j][2] << endl;
  // }  

  cv::Mat imgY= RGB2Y(img);
  cv::Mat out = cv::Mat::zeros(w,h,CV_8UC1);

  for(int j = 0;j<h;j++) {
    for(int i  = 0;i<w;i++) {
      double r = 0;
      for(int jj = -1;jj < 2;jj++) {
        for(int ii  = -1;ii < 2;ii++) {
          if( i+ii>=0 and j+jj>=0 and i+ii<w and j+jj<h) {
            r+= filter[jj+1][ii+1]*(double)imgY.at<uchar>(j+jj,i+ii);
          }
        }
      }
      r = fmax(r,0);
      r = fmin(r,255);
      out.at<uchar>(j,i) = static_cast<uchar>(r);
    }
  }
  return out;
}

uchar convertAngle(int angle)
{
  if (angle <= 22.5) {
    return 0;
  }else if (angle <= 67.5) {
    return 45;
  }else if (angle <= 112.5) {
    return 90;
  }else if (angle <= 157.5) {
    return 135;
  }
  return 0;
}

void CannyEdge(cv::Mat imgFx,cv::Mat imgFy,canny &imgCanny)
{
  int w = imgFx.cols;
  int h = imgFx.rows;
  int edge,angle;
  uchar fx,fy;

  for(int y=0;y<h;y++) {
    for(int x=0;x<w;x++) {
      fx = imgFx.at<uchar>(y,x);
      fx = fmax(fx,0.000001);
      fy = imgFy.at<uchar>(y,x);
      edge = sqrt(fx*fx+fy*fy);
      edge = min(edge,255);
      edge = max(edge,0);
      angle= atan2(fy,fx);
      angle= angle/M_PI*180;

      if(angle < -22.5){
        angle = 180 + angle;
      } else if (angle >= 157.5) {
        angle = angle - 180;
      }

      imgCanny.edge.at<uchar>(y,x) = edge;
      imgCanny.angle.at<uchar>(y,x) = convertAngle(angle);
    }
  }

}

cv::Mat answer41(cv::Mat img)
{
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  int ow = w;
  int oh = h;
  int grid = 8;
  int K = 4;
  canny c;
  c.angle = cv::Mat::zeros(h,w,CV_8UC1);
  c.edge  = cv::Mat::zeros(h,w,CV_8UC1);

  my_img<double> dct_store{w,h,ch_num};
  cv::Mat imgY= RGB2Y(img);

  imgY = Gusian(img,1.4,5);
  cv::Mat imgYx = sobal_filter(img,false);
  cv::Mat imgYy = sobal_filter(img,true);

  CannyEdge(imgYx,imgYy,c);
  
  cv::Mat disp;
  cv::Mat tmp[2]={c.edge,c.angle};
  cv::hconcat(tmp, 2, disp);

  return disp;
}

cv::Mat Canny_Nms(canny c,cv::Mat out)
{
  int w = c.angle.cols;
  int h = c.angle.rows;
  int angle;
  int edge0,edge1,edge2,maxEdge;
  int x1,y1,x2,y2;
  uchar fx,fy;

  for(int y=0;y<h;y++) {
    for(int x=0;x<w;x++) {
      angle = c.angle.at<uchar>(y,x);
      if(edge0==0) {
        x1 =x-1;y1=y;
        x2 =x+1;y2=y;
      }
      else if(edge0==45) {
        x1 =x-1;y1=y+1;
        x2 =x+1;y2=y-1;
      }
      else if(edge0==90) {
        x1 =x;y1=y-1;
        x2 =x;y2=y+1;
      }
      else if(edge0==90) {
        x1 =x-1;y1=y-1;
        x2 =x+1;y2=y+1;
      }
      x1=max(min(x1,w-1),0);y1=max(min(y1,w-1),0);
      x2=max(min(x2,w-1),0);y2=max(min(y2,w-1),0);
      edge0 = c.edge.at<uchar>(y ,x );
      edge1 = c.edge.at<uchar>(y1,x1);
      edge2 = c.edge.at<uchar>(y2,x2);
      maxEdge = max({edge0,edge1,edge2});

      edge0 = (maxEdge==edge0) ? edge0 : 0;
      out.at<uchar>(y,x) = edge0;
    }
  }

  return out;
}

cv::Mat Canny_TH(cv::Mat img,cv::Mat out,int HT=50,int LT=20)
{
  int w = img.cols;
  int h = img.rows;
  int angle;
  int edge,edgeSw;
  int x1,y1,x2,y2;
  uchar fx,fy;

  for(int y=0;y<h;y++) {
    for(int x=0;x<w;x++) {
      edge = img.at<uchar>(y,x);
      //TH
      if( edge >= HT ) {
        edge = 255;
      }
      else if( edge < LT ) {
        edge = 0;
      }
      else{
        edge = 0;
        for(int j=y-1;j<=y+1;j++) {
          for(int i=x-1;i<=x+1;i++) {
            y1 = max(min(j,h-1),0);
            x1 = max(min(i,w-1),0);
            edgeSw = img.at<uchar>(y1,x1);
            if( edgeSw >= HT) {
              edge = 255;
              break;
            }
          }
        }
      }
      out.at<uchar>(y,x) = edge;
    }
  }

  return out;
}

cv::Mat answer42(cv::Mat img)
{
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  int ow = w;
  int oh = h;
  int grid = 8;
  int K = 4;
  canny c;
  c.angle = cv::Mat::zeros(h,w,CV_8UC1);
  c.edge  = cv::Mat::zeros(h,w,CV_8UC1);

  my_img<double> dct_store{w,h,ch_num};
  cv::Mat imgY= RGB2Y(img);

  imgY = Gusian(img,1.4,5);
  cv::Mat imgYx = sobal_filter(img,false);
  cv::Mat imgYy = sobal_filter(img,true);

  CannyEdge(imgYx,imgYy,c);

  cv::Mat out = cv::Mat::zeros(h,w,CV_8UC1);
  
  out = Canny_Nms(c,out);

  cv::Mat disp;
  cv::Mat tmp[2]={c.edge,out};
  cv::hconcat(tmp, 2, disp);

  return disp;
}


cv::Mat answer43(cv::Mat img)
{
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  int ow = w;
  int oh = h;
  int grid = 8;
  int K = 4;
  canny c;
  c.angle = cv::Mat::zeros(h,w,CV_8UC1);
  c.edge  = cv::Mat::zeros(h,w,CV_8UC1);

  my_img<double> dct_store{w,h,ch_num};
  cv::Mat imgY= RGB2Y(img);

  imgY = Gusian(img,1.4,5);
  cv::Mat imgYx = sobal_filter(img,false);
  cv::Mat imgYy = sobal_filter(img,true);

  CannyEdge(imgYx,imgYy,c);

  cv::Mat out1 = cv::Mat::zeros(h,w,CV_8UC1);
  cv::Mat out2 = cv::Mat::zeros(h,w,CV_8UC1);

  out1 = Canny_Nms(c,out1);
  out2 = Canny_TH(out1,out2,50,20);

  cv::Mat disp;
  cv::Mat tmp[2]={out1,out2};
  cv::hconcat(tmp, 2, disp);

  return disp;
}

int main(int argc, const char* argv[]){
  cv::Mat img = cv::imread("../Question_41_50/imori.jpg",cv::IMREAD_COLOR);
  cv::Mat out = answer43(img);
  
  cv::imshow("sample", out);
  cv::waitKey(0);
  cv::destroyAllWindows(); 

  return 0;
}