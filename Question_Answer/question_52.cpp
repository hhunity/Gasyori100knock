#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <Eigen/Dense>
#include <chrono>

using namespace std;
using namespace cv;

using cvi = cv::Mat;

inline void img_loop(cvi &img,std::function<void(cvi&, int, int)> func){
  for(int i=0;i<img.rows;i++) {
    for(int j=0;j<img.cols;j++) {
      func(img,i,j);
    }
  }
}

cvi answer52(cvi img,cvi p_img){
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  int min_sum = std::numeric_limits<int>::max();
  double max_sum = 0;
  pair<int,int> min_p;

#if 1
    cv::Mat img_16s, img_p_16S;
    img.convertTo(img_16s, CV_16S);
    p_img.convertTo(img_p_16S, CV_16S);
    
    cv::Scalar sum;
    sum = cv::sum(img_16s);
    cv::Scalar ave    = (sum)/(img_16s.cols*img_16s.rows);
    sum = cv::sum(img_p_16S);
    cv::Scalar ave_p  = (sum)/(img_p_16S.cols*img_p_16S.rows);


    img_16s.forEach<Vec3b>([&img_16s,&img_p_16S,&max_sum,&min_p,&ave,&ave_p](Vec3b &pixcel,const int *p) {
      int y = p[0];
      int x = p[1];
      double sum = 0;
      if(y >= img_16s.rows-img_p_16S.rows or x > img_16s.cols-img_p_16S.cols) {
        return;
      }

      cvi imgRoi = img_16s(Rect(x,y,img_p_16S.cols,img_p_16S.rows));
      cv::Mat imgRoi_sub,p_img_sub,result;

      //オーバフロー防ぐため255で割ってる。
      //もしくは浮動小数点にかえる。
      //imgRoi.convertTo(imgRoiF, CV_32F);
      //p_img.convertTo(p_imgF, CV_32F);
      cv::subtract(imgRoi, ave  , imgRoi_sub);
      cv::subtract(img_p_16S    , ave_p, p_img_sub);

      cv::multiply(imgRoi_sub , p_img_sub, result, 1.0 / 255.0); 
      cv::Scalar sum_1 = cv::sum(result);

      cv::multiply(imgRoi_sub , imgRoi_sub, result, 1.0 / 255.0); 
      cv::Scalar sum_2 = cv::sum(result);

      cv::multiply(p_img_sub , p_img_sub, result, 1.0 / 255.0); 
      cv::Scalar sum_3 = cv::sum(result);
      
      for(int i = 0;i< 3 ;i++) {
        sum += ((double)sum_1[i])/(sqrt(sum_2[i])*sqrt(sum_3[i]));
      }

      if( sum > max_sum) {
        min_p.first  = x;
        min_p.second = y;
        max_sum = sum;
      }
    });
#elif 1
  img_loop(img,[&min_sum,&min_p,&p_img](cvi& img,int y,int x){
    int sum=0;
    for(int j = 0; j < p_img.rows; j++) {
      cv::Vec3b* row = p_img.ptr<cv::Vec3b>(j); // 行ごとにポインタ取得
      for(int i = 0; i < p_img.cols; i++) {
          int ix = clamp(x+i,0,img.cols-1);
          int iy = clamp(y+j,0,img.rows-1);
          sum += pow(row[j][0] - img.at<Vec3b>(iy,ix)[0],2);
          sum += pow(row[j][1] - img.at<Vec3b>(iy,ix)[1],2);
          sum += pow(row[j][2] - img.at<Vec3b>(iy,ix)[2],2);
      }
    }
    if( sum < min_sum) {
      min_p.first  = x;
      min_p.second = y;
      min_sum = sum;
    }
  });
#else
  img_loop(img,[&min_sum,&min_p,&p_img](cvi& img,int y,int x){
    int sum=0;
    img_loop(p_img,[&sum,&img,&x,&y](cvi &p_img,int yy,int xx){
      int ix = clamp(x+xx,0,img.cols-1);
      int iy = clamp(y+yy,0,img.rows-1);
      sum += pow(p_img.at<Vec3b>(yy,xx)[0] - img.at<Vec3b>(iy,ix)[0],2);
      sum += pow(p_img.at<Vec3b>(yy,xx)[1] - img.at<Vec3b>(iy,ix)[1],2);
      sum += pow(p_img.at<Vec3b>(yy,xx)[2] - img.at<Vec3b>(iy,ix)[2],2);
    });
    if( sum < min_sum) {
      min_p.first  = x;
      min_p.second = y;
      min_sum = sum;
    }
  });
#endif

  cv::rectangle(img,Point(min_p.first,min_p.second),Point(min_p.first+p_img.cols,min_p.second+p_img.rows),Scalar(0, 0, 255),2);

  return img;
}

int main(int argc, const char* argv[]){
  auto start = std::chrono::high_resolution_clock::now();

  // read image
  cv::Mat img   = cv::imread("../Question_51_60/imori.jpg", cv::IMREAD_COLOR);
  cv::Mat img_p = cv::imread("../Question_51_60/imori_part.jpg", cv::IMREAD_COLOR);

  // Canny
  cv::Mat edge = answer52(img,img_p);

  //cv::imwrite("out.jpg", out);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  cout << "time:" << duration.count()/1000 << endl;
  
  cv::imshow("answer(edge)", edge);
  cv::waitKey(0);
  cv::destroyAllWindows();
  return 0;
}