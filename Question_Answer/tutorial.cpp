#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;

int main(int argc, const char* argv[]){
  cv::Mat img = cv::imread("../tutorial/imori.jpg",cv::IMREAD_COLOR);

  int w,h;
  w = img.rows;
  h = img.cols;

  cout << w << "," << h <<endl;

  cv::Mat img2 = img.clone();

  for (int i = 0; i < w / 2; i++){
    for (int j = 0; j < h / 2; j++){
      unsigned char t = img.at<cv::Vec3b>(j,i)[0];
      img.at<cv::Vec3b>(j,i)[0] = img.at<cv::Vec3b>(j,i)[2] = 255;
      img.at<cv::Vec3b>(j,i)[2] = t;
    }
  }

  cv::Mat disp;
  cv::Mat tmp[3];
  tmp[0] = img;
  tmp[1] = cv::Mat (cv::Size(10, h), CV_8UC3, cv::Scalar(0,0,0));
  tmp[2] = img2;
  cv::hconcat(tmp, 3, disp);
  cv::imshow("sample", disp);
  cv::waitKey(0);
  cv::destroyAllWindows(); 

  cv::imwrite("out.jpg", disp);

  return 0;
}