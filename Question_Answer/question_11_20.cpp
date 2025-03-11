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


cv::Mat Avefilter(cv::Mat img,int fil_size = 3) {
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  int pad = floor(fil_size/2);

  cv::Mat out = cv::Mat::zeros(w,h,CV_8UC3);

  for(int ch = 0;ch<ch_num;ch++) {
    for(int j = 0;j<h;j++) {
      for(int i  = 0;i<w;i++) {
        int count = 0;
        int sum   = 0;
        for(int jj = -pad;jj < pad+1;jj++) {
          for(int ii  = -pad;ii < pad+1;ii++) {
            if( i+ii>=0 and j+jj>=0 and i+ii<w and j+jj<h) {
              sum += img.at<cv::Vec3b>(i+ii,j+jj)[ch];
              count ++;
            }
          }
        }
        out.at<cv::Vec3b>(i,j)[ch] = sum/count;
      }
    }
  }
  return out;
}

cv::Mat answer11(cv::Mat img) {
  return Avefilter(img);
}

cv::Mat Motionfilter(cv::Mat img,int fil_size = 3) {
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  int pad = floor(fil_size/2);
  double filter[fil_size][fil_size];

  for(int y=0;y<fil_size;y++) {
    for(int x = 0;x<fil_size;x++) {
      filter[y][x] = y==x ? 1.0f/fil_size : 0.0;
    }
  }

  cv::Mat out = cv::Mat::zeros(w,h,CV_8UC3);

  for(int ch = 0;ch<ch_num;ch++) {
    for(int j = 0;j<h;j++) {
      for(int i  = 0;i<w;i++) {
        double sum   = 0;
        for(int jj = -pad;jj < pad+1;jj++) {
          for(int ii  = -pad;ii < pad+1;ii++) {
            if( i+ii>=0 and j+jj>=0 and i+ii<w and j+jj<h) {
              sum += static_cast<double>(img.at<cv::Vec3b>(i+ii,j+jj)[ch]*filter[ii+pad][jj+pad]);
            }
          }
        }
        out.at<cv::Vec3b>(i,j)[ch] = static_cast<uchar>(sum);
      }
    }
  }
  return out;
}

cv::Mat answer12(cv::Mat img) {
  return Motionfilter(img);
}

cv::Mat MaxMinfilter(cv::Mat img,int fil_size = 3) {
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  int pad = floor(fil_size/2);
  double filter[fil_size][fil_size];

  for(int y=0;y<fil_size;y++) {
    for(int x = 0;x<fil_size;x++) {
      filter[y][x] = y==x ? 1.0f/fil_size : 0.0;
    }
  }

  cv::Mat imgY= RGB2Y(img);
  cv::Mat out = cv::Mat::zeros(w,h,CV_8UC1);


  for(int j = 0;j<h;j++) {
    for(int i  = 0;i<w;i++) {
      uchar Max = 0;
      uchar Min = 255;
      for(int jj = -pad;jj < pad+1;jj++) {
        for(int ii  = -pad;ii < pad+1;ii++) {
          if( i+ii>=0 and j+jj>=0 and i+ii<w and j+jj<h) {
            Max = max(Max,imgY.at<uchar>(i+ii,j+jj));
            Min = min(Min,imgY.at<uchar>(i+ii,j+jj)); 
          }
        }
      }
      out.at<uchar>(i,j) = Max - Min;
    }
  }
  return out;
}

cv::Mat answer13(cv::Mat img) {
  return MaxMinfilter(img);
}

cv::Mat diff_filter(cv::Mat img,bool bvertical = false) {
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  double filter[3][3]={{0,-1,0},{0,1,0},{0,0,0}};
  
  if(!bvertical){
    filter[0][1] = 0;
    filter[1][0] = -1; 
  }


  cv::Mat imgY= RGB2Y(img);
  cv::Mat out = cv::Mat::zeros(w,h,CV_8UC1);

  for(int j = 0;j<h;j++) {
    for(int i  = 0;i<w;i++) {
      double r = 0;
      for(int jj = -1;jj < 2;jj++) {
        for(int ii  = -1;ii < 2;ii++) {
          if( i+ii>=0 and j+jj>=0 and i+ii<w and j+jj<h) {
            r+= filter[jj+1][ii+1]*imgY.at<uchar>(j+jj,i+ii);
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

cv::Mat answer14(cv::Mat img) {
  cv::Mat vimg = diff_filter(img,true);
  cv::Mat himg = diff_filter(img,false);
  cv::Mat disp;
  cv::Mat tmp[2]={vimg,himg};
  cv::hconcat(tmp, 2, disp);

  return disp;
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

cv::Mat answer15(cv::Mat img) {
  cv::Mat vimg = sobal_filter(img,true);
  cv::Mat himg = sobal_filter(img,false);
  cv::Mat disp;
  cv::Mat tmp[2]={vimg,himg};
  cv::hconcat(tmp, 2, disp);

  return disp;
}

cv::Mat preitt_filter(cv::Mat img,bool bvertical = false) {
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  double  filter[3][3]={{-1,-1,-1},{0,0,0},{1,1,1}};
  double hfilter[3][3]={{-1, 0, 1},{-1,0,1},{-1,0,1}};

  if (!bvertical){
    for(int y=0;y< 3;y++) {
      copy(hfilter[y],hfilter[y]+3,filter[y]);
    }
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

cv::Mat answer16(cv::Mat img) {
  cv::Mat vimg = preitt_filter(img,true);
  cv::Mat himg = preitt_filter(img,false);
  cv::Mat disp;
  cv::Mat tmp[2]={vimg,himg};
  cv::hconcat(tmp, 2, disp);

  return disp;
}

cv::Mat laplacian_filter(cv::Mat img) {
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  double  filter[3][3]={{0, 1, 0}, {1, -4, 1}, {0, 1, 0}};

  // for(int j = 0;j<3;j++) {
  //   cout << filter[j][0] << "," << filter[j][1] << "," << filter[j][2] << endl;
  // }  
  cv::Mat imgY;
  if(ch_num!=1) {
    imgY= RGB2Y(img);
  }else{
    imgY = img;
  }
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

cv::Mat answer17(cv::Mat img) {
  cv::Mat vimg = laplacian_filter(img);
  return vimg;
}

cv::Mat emboss_filter(cv::Mat img) {
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  double  filter[3][3]={{-2, -1, 0}, {-1, 1, 1}, {0, 1, 2}};
  
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

cv::Mat answer18(cv::Mat img) {
  cv::Mat vimg = emboss_filter(img);
  return vimg;
}

cv::Mat log_filter(cv::Mat img,int fil_size,double sigma) {
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  int pad = floor(fil_size/2);
  double filter[fil_size][fil_size];
  double fil_sum =0;
  double r = 0;

  cv::Mat imgY= RGB2Y(img);
  cv::Mat out = cv::Mat::zeros(w,h,CV_8UC1);

  for(int i =0;i< fil_size;i++) {
    for(int j =0;j<fil_size;j++) {
      int x = j - pad;
      int y = i - pad;
      filter[i][j] = (x * x + y * y - sigma * sigma) / (2 * M_PI * pow(sigma, 6)) * exp( - (x * x + y * y) / (2 * sigma * sigma));
      fil_sum+=filter[i][j];
    }
  }

  for(int i =0;i< fil_size;i++) {
    for(int j =0;j<fil_size;j++) {
      filter[i][j] /=fil_sum;
    }
  }

  for(int j = 0;j<h;j++) {
    for(int i  = 0;i<w;i++) {
      r = 0;
      for(int jj = -pad;jj < pad+1;jj++) {
        for(int ii  = -pad;ii < pad +1;ii++) {
          if( i+ii>=0 and j+jj>=0 and i+ii<w or j+jj<h) {
            r += filter[ii+pad][jj+pad] * imgY.at<uchar>(j+jj,i+ii);
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

cv::Mat answer19(cv::Mat img) {
  cv::Mat vimg = log_filter(img,5,3);
  return vimg;
}


int main(int argc, const char* argv[]){
  cv::Mat img = cv::imread("../Question_11_20/imori.jpg",cv::IMREAD_COLOR);
  cv::Mat out = answer19(img);
  cv::imshow("sample", out);
  cv::waitKey(0);
  cv::destroyAllWindows(); 

  return 0;
}