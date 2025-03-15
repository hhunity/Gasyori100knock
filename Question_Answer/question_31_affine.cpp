#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <algorithm>
#include <Eigen/Dense>

using namespace std;

cv::Mat affine(cv::Mat img, double a, double b, double c, double d, double tx, double ty)
{
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  int ow = a*w;
  int oh = d*h;

  cv::Mat out = cv::Mat::zeros(oh,ow,CV_8UC3);

  cout << a << "," << b <<  "," << c << "," << d << "," << tx << "," << ty << endl;

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

cv::Mat answer30(cv::Mat img)
{
  Eigen::Matrix3d A;
  Eigen::Matrix3d Ainv;
  Eigen::Vector3d B;
  Eigen::Matrix3d C;
  Eigen::Vector3d D;

  double a = 30.0/img.rows;
  double b = 30.0/img.cols;

  A << 1 , a , 0,
       b , 1 , 0,
       0 , 0 , 1;
  B << img.cols/2.,
       img.rows/2.,
       1;
  
  Ainv = A.inverse().eval();
  D =  Ainv * B;
  D = (D - B).eval();

  cout << Ainv << endl;
  
  return affine(img,A(0,0),A(0,1),A(1,0),A(1,1),D(0),D(1));
}

#include <cblas.h>
#include <lapacke.h>

cv::Mat answer30_Blas(cv::Mat img)
{
  double a = 30.0/img.rows;
  double b = 30.0/img.cols;
  double A[9] =  {1.0,a  ,0.0,
                  b  ,1.0,0.0,
                  0.0,0.0,1.0};

  double B[3] = { img.cols/2.,
                  img.rows/2.,
                  1.0};

  double C[3] = {0.0};

  // LU分解
  // double A[9] = {1.0, 2.0, 3.0,
  // 0.0, 1.0, 4.0,
  // 5.0, 6.0, 0.0};
  // 行列 A の逆行列を格納するための配列
  double A_inv[9];
  std::copy(A, A+9, A_inv);  // 元の行列をコピー

  int N = 3;      // 行列の次元
  int pivots[3];  // ピボットの配列
  int info;       // 戻り値

  //LU分解
  info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, N, N, A_inv, N, pivots);
  if (info != 0) {
      std::cerr << "LU decomposition failed!" << std::endl;
      throw 0;
  }

  // 逆行列の計算
  info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, N, A_inv, N, pivots);
  if (info != 0) {
      std::cerr << "Matrix inversion failed!" << std::endl;
      throw 0;
  }

  // 結果を表示
  std::cout << "A_inv (逆行列):" << std::endl;
  for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
          std::cout << A_inv[i * N + j] << " ";
      }
      std::cout << std::endl;
  }

  // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 2, 1, 2, 1.0, &A[0][0], 2, &B[0][0], 2, 0.0, &C[0][0], 2);
  //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2, 1, 2, 1.0, &A[0][0], 2, &B[0][0], 1, 0.0, &C[0][0], 1);

  //C =  Ainv * B;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 1, 3, 1.0, A_inv, 3, B, 1, 0.0, C, 1);

  // ベクトルの引き算: A - B
  // C = C + (-1) * B
  cblas_daxpy(3, -1.0, B, 1, C, 1);  // C = C + (-1) * B

  // 結果表示
  std::cout << "C = A * B: " << std::endl;
  for(int i = 0; i < 3; i++) {
      std::cout << C[i] << std::endl; // Cは3x1の行列になるので1列だけ表示
  }

  return affine(img,A[0],A[1],A[3*1],A[3*1+1],C[0],C[1]);
}


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
cv::Mat answer32(cv::Mat img)
{
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  int ow = w;
  int oh = h;
  cv::Mat imgY = RGB2Y(img);

  for(int k=0;k<w;k++) {
    for(int l=0;l<h;l++) {
      
    }
  }

}

int main(int argc, const char* argv[]){
  cv::Mat img = cv::imread("../Question_21_30/imori.jpg",cv::IMREAD_COLOR);
  cv::Mat out = answer30_Blas(img);
  cv::imshow("sample", out);
  cv::waitKey(0);
  cv::destroyAllWindows(); 

  return 0;
}