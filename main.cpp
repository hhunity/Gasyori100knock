#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;

int main(int argc, const char* argv[]){
  cv::Mat redImg(cv::Size(320, 240), CV_8UC3, cv::Scalar(0, 0, 255));
  cv::namedWindow("red", cv::WINDOW_AUTOSIZE);
  
  cv::imshow("red", redImg);
  cv::waitKey(0);

  cv::destroyAllWindows();
  
  return 0;
}