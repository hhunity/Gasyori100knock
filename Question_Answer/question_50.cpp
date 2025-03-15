#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <algorithm>
#include <Eigen/Dense>
using namespace std;

// RGB to Gray scale
cv::Mat BGR2GRAY(cv::Mat img){
  // get height and width
  int height = img.rows;
  int width = img.cols;
  int channel = img.channels();

  // prepare output
  cv::Mat out = cv::Mat::zeros(height, width, CV_8UC1);
  
  // BGR -> Gray
  for (int y = 0; y < height; y++){
    for (int x = 0; x < width; x++){
      out.at<uchar>(y, x) = (int)((float)img.at<cv::Vec3b>(y, x)[0] * 0.0722 + \
				  (float)img.at<cv::Vec3b>(y, x)[1] * 0.7152 + \
				  (float)img.at<cv::Vec3b>(y, x)[2] * 0.2126);
    }
  }
  return out;
}

float clip(float value, float min, float max){
  return fmin(fmax(value, 0), 255);
}

// gaussian filter
cv::Mat gaussian_filter(cv::Mat img, double sigma, int kernel_size){
  int height = img.rows;
  int width = img.cols;
  int channel = img.channels();

  // prepare output
  cv::Mat out = cv::Mat::zeros(height, width, CV_8UC3);
  if (channel == 1) {
    out = cv::Mat::zeros(height, width, CV_8UC1);
  }

  // prepare kernel
  int pad = floor(kernel_size / 2);
  int _x = 0, _y = 0;
  double kernel_sum = 0;
  
  // get gaussian kernel
  float kernel[kernel_size][kernel_size];

  for (int y = 0; y < kernel_size; y++){
    for (int x = 0; x < kernel_size; x++){
      _y = y - pad;
      _x = x - pad; 
      kernel[y][x] = 1 / (2 * M_PI * sigma * sigma) * exp( - (_x * _x + _y * _y) / (2 * sigma * sigma));
      kernel_sum += kernel[y][x];
    }
  }

  for (int y = 0; y < kernel_size; y++){
    for (int x = 0; x < kernel_size; x++){
      kernel[y][x] /= kernel_sum;
    }
  }

  // filtering
  double v = 0;
  
  for (int y = 0; y < height; y++){
    for (int x = 0; x < width; x++){
      // for BGR
      if (channel == 3){
        for (int c = 0; c < channel; c++){
          v = 0;
          for (int dy = -pad; dy < pad + 1; dy++){
            for (int dx = -pad; dx < pad + 1; dx++){
              if (((x + dx) >= 0) && ((y + dy) >= 0) && ((x + dx) < width) && ((y + dy) < height)){
                v += (double)img.at<cv::Vec3b>(y + dy, x + dx)[c] * kernel[dy + pad][dx + pad];
              }
            }
          }
          out.at<cv::Vec3b>(y, x)[c] = (uchar)clip(v, 0, 255);
        }
      } else {
        // for Gray
        v = 0;
        for (int dy = -pad; dy < pad + 1; dy++){
          for (int dx = -pad; dx < pad + 1; dx++){
            if (((x + dx) >= 0) && ((y + dy) >= 0) && ((x + dx) < width) && ((y + dy) < height)){
              v += (double)img.at<uchar>(y + dy, x + dx) * kernel[dy + pad][dx + pad];
            }
          }
        }
        out.at<uchar>(y, x) = (uchar)clip(v, 0, 255);
      }
    }
  }
  return out;
}

// Sobel filter
cv::Mat sobel_filter(cv::Mat img, int kernel_size, bool horizontal){
  int height = img.rows;
  int width = img.cols;
  int channel = img.channels();

  // prepare output
  cv::Mat out = cv::Mat::zeros(height, width, CV_8UC1);

  // prepare kernel
  double kernel[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

  if (horizontal){
    kernel[0][1] = 0;
    kernel[0][2] = -1;
    kernel[1][0] = 2;
    kernel[1][2] = -2;
    kernel[2][0] = 1;
    kernel[2][1] = 0;
  }

  int pad = floor(3 / 2);

  double v = 0;

  // filtering  
  for (int y = 0; y < height; y++){
    for (int x = 0; x < width; x++){
      v = 0;
      for (int dy = -pad; dy < pad + 1; dy++){
        for (int dx = -pad; dx < pad + 1; dx++){
          if (((y + dy) >= 0) && (( x + dx) >= 0) && ((y + dy) < height) && ((x + dx) < width)){
            v += (double)img.at<uchar>(y + dy, x + dx) * kernel[dy + pad][dx + pad];
          }
        }
      }
      out.at<uchar>(y, x) = (uchar)clip(v, 0, 255);
    }
  }
  return out;
}

// get edge
cv::Mat get_edge(cv::Mat fx, cv::Mat fy){
  // get height and width
  int height = fx.rows;
  int width = fx.cols;

  // prepare output
  cv::Mat out = cv::Mat::zeros(height, width, CV_8UC1);

  double _fx, _fy;

  for(int y = 0; y < height; y++){
    for(int x = 0; x < width; x++){
      _fx = (double)fx.at<uchar>(y, x);
      _fy = (double)fy.at<uchar>(y, x);

      out.at<uchar>(y, x) = (uchar)clip(sqrt(_fx * _fx + _fy * _fy), 0, 255);
    }
  }

  return out;
}

// get angle
cv::Mat get_angle(cv::Mat fx, cv::Mat fy){
  // get height and width
  int height = fx.rows;
  int width = fx.cols;

  // prepare output
  cv::Mat out = cv::Mat::zeros(height, width, CV_8UC1);

  double _fx, _fy;
  double angle;

  for(int y = 0; y < height; y++){
    for(int x = 0; x < width; x++){
      _fx = fmax((double)fx.at<uchar>(y, x), 0.000001);
      _fy = (double)fy.at<uchar>(y, x);

      angle = atan2(_fy, _fx);
      angle = angle / M_PI * 180;

      if(angle < -22.5){
        angle = 180 + angle;
      } else if (angle >= 157.5) {
        angle = angle - 180;
      }

      //std::cout << angle << " " ;

      // quantization
      if (angle <= 22.5){
        out.at<uchar>(y, x) = 0;
      } else if (angle <= 67.5){
        out.at<uchar>(y, x) = 45;
      } else if (angle <= 112.5){
        out.at<uchar>(y, x) = 90;
      } else {
        out.at<uchar>(y, x) = 135;
      }
    }
  }

  return out;
}


// non maximum suppression
cv::Mat non_maximum_suppression(cv::Mat angle, cv::Mat edge){
  int height = angle.rows;
  int width = angle.cols;
  int channel = angle.channels();

  int dx1, dx2, dy1, dy2;
  int now_angle;

  // prepare output
  cv::Mat _edge = cv::Mat::zeros(height, width, CV_8UC1);

  for (int y = 0; y < height; y++){
    for (int x = 0; x < width; x++){
      now_angle = angle.at<uchar>(y, x);
      // angle condition
      if (now_angle == 0){
        dx1 = -1;
        dy1 = 0;
        dx2 = 1;
        dy2 = 0;
      } else if(now_angle == 45) {
        dx1 = -1;
        dy1 = 1;
        dx2 = 1;
        dy2 = -1;
      } else if(now_angle == 90){
        dx1 = 0;
        dy1 = -1;
        dx2 = 0;
        dy2 = 1;
      } else {
        dx1 = -1;
        dy1 = -1;
        dx2 = 1;
        dy2 = 1;
      }

      if (x == 0){
        dx1 = fmax(dx1, 0);
        dx2 = fmax(dx2, 0);
      }
      if (x == (width - 1)){
        dx1 = fmin(dx1, 0);
        dx2 = fmin(dx2, 0);
      }
      if (y == 0){
        dy1 = fmax(dy1, 0);
        dy2 = fmax(dy2, 0);
      }
      if (y == (height - 1)){
        dy1 = fmin(dy1, 0);
        dy2 = fmin(dy2, 0);
      }

      // if pixel is max among adjuscent pixels, pixel is kept
      if (fmax(fmax(edge.at<uchar>(y, x), edge.at<uchar>(y + dy1, x + dx1)), edge.at<uchar>(y + dy2, x + dx2)) == edge.at<uchar>(y, x)) {
        _edge.at<uchar>(y, x) = edge.at<uchar>(y, x);
      }
    }
  }

  return _edge;
}

// histerisis
cv::Mat histerisis(cv::Mat edge, int HT, int LT){
  int height = edge.rows;
  int width = edge.cols;
  int channle = edge.channels();

  // prepare output
  cv::Mat _edge = cv::Mat::zeros(height, width, CV_8UC1);

  int now_pixel;

  for (int y = 0; y < height; y++){
    for (int x = 0; x < width; x++){
      // get pixel
      now_pixel = edge.at<uchar>(y, x);

      // if pixel >= HT
      if (now_pixel >= HT){
        _edge.at<uchar>(y, x) = 255;
      } 
      // if LT < pixel < HT
      else if (now_pixel > LT) {
        for (int dy = -1; dy < 2; dy++){
          for (int dx = -1; dx < 2; dx++){
            // if 8 nearest neighbor pixel >= HT
            if (edge.at<uchar>(fmin(fmax(y + dy, 0), 255), fmin(fmax(x + dx, 0), 255)) >= HT){
              _edge.at<uchar>(y, x) = 255;
            }
          }
        }
      }
    }
  }
  return _edge;
}


// Canny
cv::Mat Canny(cv::Mat img){
  // BGR -> Gray
  cv::Mat gray = BGR2GRAY(img);

  // gaussian filter
  cv::Mat gaussian = gaussian_filter(gray, 1.4, 5);

  // sobel filter (vertical)
  cv::Mat fy = sobel_filter(gaussian, 3, false);

  // sobel filter (horizontal)
  cv::Mat fx = sobel_filter(gaussian, 3, true);

  // get edge
  cv::Mat edge = get_edge(fx, fy);

  // get angle
  cv::Mat angle = get_angle(fx, fy);

  // edge non-maximum suppression
  edge = non_maximum_suppression(angle, edge);

  // histerisis
  edge = histerisis(edge, 50, 20);

  return edge;
}

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

cv::Mat Y_Th2(cv::Mat img, int th){

  int w = img.cols;
  int h = img.rows;
  uchar gray;

  for(int y = 0;y<h;y++) {
    for(int x  = 0;x<w;x++) {
      gray = img.at<uchar>(y,x);
      img.at<uchar>(y,x) = gray < th ? 0 : 255;
    
    }
  }

  return img;
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
    for(int y = 0;y < h;y++) {
      for(int x = 0;x < w;x++) {
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

    // cout << "Sb2 :" << Sb2 << "," << W0 << "," << W1 <<"," << W0 + W1 << "," << M0 <<"," << M1  << endl;

    return Sb2;
}

cv::Mat OOTH(cv::Mat img)
{
  float   W0,W1;
  float   M0,M1;
  int     Sb2,maxSb2 = 0,th = 0;

  for(int i =0 ;i<255;i++) {
    Sb2 = calc_Sb2(img,i,W0,W1,M0,M1);
    if(Sb2 > maxSb2) {
      maxSb2 = Sb2;
      th = i;
    }
  }

  cout << "maxSb2 :" << maxSb2 << " th =" << th << endl;

  return Y_Th2(img,th);
}

cv::Mat MorphologicalProcess(cv::Mat img,bool (*comp)(int))
{
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  cv::Mat out = img.clone();
  int fil[3][3] ={{0,1,0},{1,0,1},{0,1,0}};
  int val;

  for(int y = 0;y < h;y++) {
    for(int x = 0;x < w;x++) {
      val = 0;
      // if(img.at<uchar>(y,x)!=255) {
      //   out.at<uchar>(y,x) = 0;
      //   continue;
      // };
      for(int yy=0;yy<3;yy++) {
        for(int xx=0;xx<3;xx++) {
          int iy = fmin(fmax(y+(yy-1),0),h-1);
          int ix = fmin(fmax(x+(xx-1),0),w-1);
          // cout << iy << "," << ix << endl;
          val += img.at<uchar>(iy,ix)*fil[yy][xx];
        }
      }
      // cout << val << "," << 255*4 << endl;
      if(comp(val)) {
        out.at<uchar>(y,x) = 0;
      }
    }
  }
  
  return out;
}
cv::Mat MorphologicalDilation(cv::Mat img)
{
  return MorphologicalProcess(img ,[](int a){return a < 255;});
}
cv::Mat MorphologicalErogion(cv::Mat img)
{
  return MorphologicalProcess(img ,[](int a){return a < 255*4;});
}

cv::Mat answer50(cv::Mat img)
{
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  int ow = w;
  int oh = h;

  // cv::Mat imgY    = RGB2Y(img);
  // cv::Mat imgY_th = OOTH(imgY);
  cv::Mat imgCay  = Canny(img);
  cv::Mat imgY_mo1= MorphologicalDilation(imgCay);
  cv::Mat imgY_mo2= MorphologicalDilation(imgY_mo1);

  return imgY_mo2;
}

int main(int argc, const char* argv[]){
  // read image
  cv::Mat img = cv::imread("../Question_41_50/imori.jpg", cv::IMREAD_COLOR);

  // Canny
  cv::Mat edge = answer50(img);

  //cv::imwrite("out.jpg", out);
  cv::imshow("answer(edge)", edge);
  cv::waitKey(0);
  cv::destroyAllWindows();

  return 0;
}