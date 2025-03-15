#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>


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
  edge = histerisis(edge, 100, 30);

  return edge;
}

cv::Mat hough_conversion(cv::Mat img)
{
  int height  = img.rows;
  int width   = img.cols;
  int channle = img.channels();
  int rmax = std::sqrt(height*height + width * width);
  double angle;

  std::cout << rmax << std::endl;

  cv::Mat hough = cv::Mat::zeros(2*rmax, 180, CV_8UC1);

  for(int y=0;y<height;y++) {
    for(int x=0;x<width;x++) {
      if( img.at<uchar>(y,x) != 255 ) {
        continue;
      }
      for(int t=0;t<180;t++) {
        angle = M_PI/180*t;
        double rho = x*std::cos(angle)+y*std::sin(angle);
        hough.at<uchar>(rho+rmax,t) += 1;
      }
    }
  }

  return hough;

}

cv::Mat hough_nms(cv:: Mat img)
{
  int height  = img.rows;
  int width   = img.cols;
  int channle = img.channels();
  uchar val,val2;

  cv::Mat out = cv::Mat::zeros(height, width, CV_8UC1);

  for(int y=0;y<height;y++) {
    for(int x=0;x<width;x++) {
      val = img.at<uchar>(y,x);
      val2= val;
      if(val==0) {
        continue;
      }

      for(int j=-1;j<=1;j++) {
        for(int i=-1;i<=1;i++) {
          int jj = fmax(fmin(y+j,height-1),0);
          int ii = fmax(fmin(x+i,width-1 ),0);
          if(val<img.at<uchar>(jj,ii) and ((j!=0) and (i!=0)) ) {
            val2 = 0;
          }
        }
      }
      out.at<uchar>(y,x) = val2;
    }
  }

  uchar hist[256] = {};

  for(int y=0;y<height;y++) {
    for(int x=0;x<width;x++) {
      val = out.at<uchar>(y,x);
      hist[val]=hist[val]+1;
    }
  }

  int i,count = 0;

  for(i = 255;i>=0;i--) {
    count += hist[i];
    if(count >= 20) {
      break;
    }
  }

  int count_th = i;

  for(int y=0;y<height;y++) {
    for(int x=0;x<width;x++) {
      val = out.at<uchar>(y,x);
      if( val <= count_th) {
        out.at<uchar>(y,x) = 0;
      }else {
        out.at<uchar>(y,x) = 255;
      }
    }
  }


  return out;
}

cv::Mat hough_inverse(cv::Mat img,cv::Mat hough)
{
  int height = img.rows;
  int width  = img.cols;
  int max_r  = hough.rows;
  int max_t  = hough.cols;
  double angle;
  double _cos,_sin;

  for(int r=0;r<max_r;r++) {
    for(int t=0;t<max_t;t++) {
      if( hough.at<uchar>(r,t) != 255 ) {
        continue;
      }
      _cos = cos(t*M_PI/180);
      _sin = sin(t*M_PI/180);

      if((_cos==0) or (_sin==0)) {
        continue;
      }

      for(int x =0 ;x<width;x++) {
        int y = -_cos/_sin * x + (r-max_r/2)/_sin;
        if((y>=0) and (y<height)) {
          img.at<cv::Vec3b>(y,x) = cv::Vec3b(0,0,255);
        }
      }
      for(int y =0 ;y<height;y++) {
        int x = -_sin/_cos * y + (r-max_r/2)/_cos;
        if((x>=0) and (x<width)) {
          img.at<cv::Vec3b>(y,x) = cv::Vec3b(0,0,255);
        }
      }
    }
  }

  return img;

}

cv::Mat get_line1(cv::Mat img)
{
  cv::Mat img_canny = Canny(img);

  cv::Mat img_hough = hough_conversion(img_canny);

  return img_hough;
}

cv::Mat get_line2(cv::Mat img)
{
  cv::Mat img_canny = Canny(img);

  cv::Mat img_hough = hough_conversion(img_canny);

  cv::Mat img_nms = hough_nms(img_hough);

  return img_nms;
}

cv::Mat get_line3(cv::Mat img)
{
  cv::Mat img_canny = Canny(img);

  cv::Mat img_hough = hough_conversion(img_canny);

  cv::Mat img_nms = hough_nms(img_hough);

  cv::Mat img_out = hough_inverse(img,img_nms);

  return img_out;
}

int main(int argc, const char* argv[]){
  // read image
  cv::Mat img = cv::imread("../Question_41_50/thorino.jpg", cv::IMREAD_COLOR);

  // Canny
  cv::Mat edge = get_line3(img);

  //cv::imwrite("out.jpg", out);
  cv::imshow("answer(edge)", edge);
  cv::waitKey(0);
  cv::destroyAllWindows();

  return 0;
}