#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <cblas.h>  // BLAS のヘッダー
#include <lapacke.h>

using namespace std;

int main(int argc, const char* argv[]){
  // cv::Mat redImg(cv::Size(320, 240), CV_8UC3, cv::Scalar(0, 0, 255));
  // cv::namedWindow("red", cv::WINDOW_AUTOSIZE);
  
  // cv::imshow("red", redImg);
  // cv::waitKey(0);

  // cv::destroyAllWindows();

  // Eigen::Matrix2d A;
  // A << 3, -1,
  //     -1, 3;
  
  // Eigen::EigenSolver<Eigen::Matrix2d> solver(A);
  // std::cout << "Eigenvalues:\n" << solver.eigenvalues() << "\n";
  // std::cout << "Eigenvectors:\n" << solver.eigenvectors() << "\n";

  // Eigen::Matrix3d A;
  // Eigen::Vector3d B;
  // Eigen::Matrix3d C;
  // Eigen::Vector3d D;

  // double a = 30;
  // A << 1 , a , 0,
  //      0 , 1 , 0,
  //      0 , 0 , 1;
  // B << 1/2.,
  //      1/2.,
  //      1;

  // A = A.inverse().eval();
  // D = A * B;
  // D = (D - B).eval();

  // cout << D << endl;

  // double A[2][2] = {{1.0,2.0},{3.0,4.0}};
  // double B[2][1] = {{5.0},{6.0}};
  // double C[2][2] = {1.0};

  // // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 2, 1, 2, 1.0, &A[0][0], 2, &B[0][0], 2, 0.0, &C[0][0], 2);
  // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2, 1, 2, 1.0, &A[0][0], 2, &B[0][0], 1, 0.0, &C[0][0], 1);

  // // 結果表示
  // std::cout << "C = A * B: " << std::endl;
  // for(int i = 0; i < 2; i++) {
  //     for(int j = 0; j < 2; j++) {
  //         std::cout << C[i][j] << " ";
  //     }
  //     std::cout << std::endl;
  // }

  // LU分解
  double A[9] = {1.0, 2.0, 3.0,
    0.0, 1.0, 4.0,
    5.0, 6.0, 0.0};

  // 行列 A の逆行列を格納するための配列
  double A_inv[9];
  std::copy(A, A+9, A_inv);  // 元の行列をコピー

  int N = 3;  // 行列の次元
  int pivots[3];  // ピボットの配列
  int info;  // 戻り値
  info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, N, N, A_inv, N, pivots);
  if (info != 0) {
      std::cerr << "LU decomposition failed!" << std::endl;
      return -1;
  }

  // 逆行列の計算
  info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, N, A_inv, N, pivots);
  if (info != 0) {
      std::cerr << "Matrix inversion failed!" << std::endl;
      return -1;
  }

  // 結果を表示
  std::cout << "A_inv (逆行列):" << std::endl;
  for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
          std::cout << A_inv[i * N + j] << " ";
      }
      std::cout << std::endl;
  }

  return 0;
}