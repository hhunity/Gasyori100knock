#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <algorithm>

using namespace std;



cv::Mat answer01(){
  cv::Mat img = cv::imread("../tutorial/imori.jpg",cv::IMREAD_COLOR);

  int w = img.cols;
  int h = img.rows;

  for(int j = 0;j<h;j++) {
    for(int i  = 0;i<w;i++) {
      unsigned char t = img.at<cv::Vec3b>(j,i)[0]; 
      img.at<cv::Vec3b>(j,i)[0] =  img.at<cv::Vec3b>(j,i)[2];
    }
  }

  return img;

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

cv::Mat RGB2Y(){

  cv::Mat img = cv::imread("../tutorial/imori.jpg",cv::IMREAD_COLOR);
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
cv::Mat answer02(){
  return RGB2Y();
}

cv::Mat Y_Th(int th){

  cv::Mat img = cv::imread("../tutorial/imori.jpg",cv::IMREAD_COLOR);
  int w = img.cols;
  int h = img.rows;
  cv::Mat out = cv::Mat::zeros(h,w,CV_8UC1);

  uchar gray;

  for(int y = 0;y<h;y++) {
    for(int x  = 0;x<w;x++) {
      gray = 0.2126 * (float)img.at<cv::Vec3b>(y,x)[2] + \
                                   0.7152 * (float)img.at<cv::Vec3b>(y,x)[1] + \
                                   0.0722 * (float)img.at<cv::Vec3b>(y,x)[0];
      out.at<uchar>(y,x) = gray < th ? 0 : 255;
    
    }
  }

  return out;
}

cv::Mat answer03(){
  return Y_Th(128);
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

    cout << "Sb2 :" << Sb2 << "," << W0 << "," << W1 <<"," << W0 + W1 << "," << M0 <<"," << M1  << endl;

    return Sb2;
}

cv::Mat OOTH(){

  cv::Mat img = RGB2Y();
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

  return Y_Th(th);
}

cv::Mat answer04(){
  return OOTH();
}

void RGB2HSV(uchar R,uchar G,uchar B,float& H,float& V,float& S)
{
  uchar Max = max({B,G,R});
  uchar Min = min({B,G,R});

  if(Min==Max) {
    H = 0;
  }else if(Min == B) {
    H=(float)60*(G-R)/(Max-Min) + 60;
  }else if(Min == R) {
    H=(float)60*(B-G)/(Max-Min) + 180;
  }else {
    H=(float)60*(R-B)/(Max-Min) + 300;
  }
  V = (float)Max/255;
  S = (float)Max/255 -(float)Min/255 ;

  H = H >360 ? 360 : H;
  V = V >1   ? 0.99: V;
  S = S >1   ? 0.99: S;

  H = H <0 ? 0: H;
  V = V <0 ? 0: V;
  S = S <0 ? 0: S;

  // cout << H << "," << S << "," << V << endl;
}

void HSV2RGB(float H,float V,float S,uchar& oR,uchar& oG,uchar& oB)
{
  float C = S;
  float H2 = H /60;
  // float X = C * (1 -abs( ((float)(((int)(H2*1000))%2))/1000 - 1));
  float X = C * (1 -abs( fmod(H2,2) - 1));
  float R,G,B;

  cout << C << "," << X << "," <<H2 << endl;


  R=G=B=(V-C);

  if( H2 < 1 ) {
    R += C;
    G += X;
    B += 0;
  }
  else if(H2<2) {
    R += X;
    G += C;
    B += 0;  
  }
  else if(H2<3) {
    R += 0;
    G += C;
    B += X;
  }
  else if(H2<4) {
    R += 0;
    G += X;
    B += C;
  }
  else if(H2 < 5){
    R += X;
    G += 0;
    B += C;
  }
  else if(H2 < 6){
    R += C;
    G += 0;
    B += X;
  }else{
    R += 0;
    G += 0;
    B += 0;
  }

  cout << R << "," << G << "," << B << endl;

  R*=255;
  G*=255;
  B*=255;

  R = R>255 ? 255 : R; R = R<0 ? 0 : R;
  G = G>255 ? 255 : G; G = G<0 ? 0 : G;
  B = B>255 ? 255 : B; B = B<0 ? 0 : B;

  oR=R;
  oG=G;
  oB=B;
}

cv::Mat answer05() {
  cv::Mat img = cv::imread("../tutorial/imori.jpg",cv::IMREAD_COLOR);
  int w=img.rows;
  int h=img.cols;

  for(int y=0;y<h;y++) {
    for(int x = 0;x<w;x++){
      uchar B = img.at<cv::Vec3b>(y,x)[0];
      uchar G = img.at<cv::Vec3b>(y,x)[1];
      uchar R = img.at<cv::Vec3b>(y,x)[2];
      float   H,S,V;
      RGB2HSV(R,G,B,H,S,V);
      
      H=fmod(H+180,360);
      // H+=180;
      // if(H>=360){H-=360;}
      
      HSV2RGB(H,S,V,R,G,B);
    
      img.at<cv::Vec3b>(y,x)[0] = B;
      img.at<cv::Vec3b>(y,x)[1] = G;
      img.at<cv::Vec3b>(y,x)[2] = R;
    }
  }

  return img;

}

uchar cut_color(uchar a) {
  uchar result;

  if(  a < 64) {
    result = 32;
  }
  else if(a < 128) {
    result = 96;
  }
  else if(a < 192) {
    result = 160;
  }
  else  {
    result = 224;
  }

  return result;

}

cv::Mat answer06() {
  cv::Mat img = cv::imread("../tutorial/imori.jpg",cv::IMREAD_COLOR);
  int w=img.rows;
  int h=img.cols;

  for(int y=0;y<h;y++) {
    for(int x = 0;x<w;x++){
      uchar B = img.at<cv::Vec3b>(y,x)[0];
      uchar G = img.at<cv::Vec3b>(y,x)[1];
      uchar R = img.at<cv::Vec3b>(y,x)[2];
      
      img.at<cv::Vec3b>(y,x)[0] = cut_color(B);
      img.at<cv::Vec3b>(y,x)[1] = cut_color(G);
      img.at<cv::Vec3b>(y,x)[2] = cut_color(R);
    }
  }

  return img;

}

cv::Mat Pooling(int div = 8,bool bmax=false) {
  cv::Mat img = cv::imread("../tutorial/imori.jpg",cv::IMREAD_COLOR);
  int w=img.rows;
  int h=img.cols;
  int dw = w/div;
  int dh = h/div;
  int sumR,sumG,sumB;

  for(int y=0;y<dh;y++) {
    for(int x = 0;x<dw;x++){
      sumR=sumB=sumG=0;
      
      for(int yy=y*div; yy < (y+1)*div ; yy++) {
        for(int xx=x*div; xx < (x+1)*div ; xx++) {
          if(bmax) {
            sumR= max((int)img.at<cv::Vec3b>(yy,xx)[2],sumR);
            sumG= max((int)img.at<cv::Vec3b>(yy,xx)[1],sumG);
            sumB= max((int)img.at<cv::Vec3b>(yy,xx)[0],sumB);
          }else{
            sumR+=img.at<cv::Vec3b>(yy,xx)[2];
            sumG+=img.at<cv::Vec3b>(yy,xx)[1];
            sumB+=img.at<cv::Vec3b>(yy,xx)[0];
          }
        }
      }
      if(!bmax) {
        sumR/=(div*div);
        sumG/=(div*div);
        sumB/=(div*div);
      }
      for(int yy=y*div; yy < (y+1)*div ; yy++) {
        for(int xx=x*div; xx < (x+1)*div ; xx++) {
          img.at<cv::Vec3b>(yy,xx)[2] = sumR;
          img.at<cv::Vec3b>(yy,xx)[1] = sumG;
          img.at<cv::Vec3b>(yy,xx)[0] = sumB;
        }
      }
    }
  }

  return img;

}
cv::Mat answer07() {
  return Pooling(8);
}
cv::Mat answer08() {
  return Pooling(8,true);
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

cv::Mat answer09(cv::Mat img) {
  return Gusian(img);
}

cv::Mat Median(cv::Mat img,int fil_size = 3) {
  int w = img.cols;
  int h = img.rows;
  int ch_num = img.channels();
  int r[fil_size*fil_size];
  int pad = floor(fil_size/2);

  cv::Mat out = cv::Mat::zeros(w,h,CV_8UC3);

  for(int ch = 0;ch<ch_num;ch++) {
    for(int j = 0;j<h;j++) {
      for(int i  = 0;i<w;i++) {
        int count = 0;
        for(int jj = -pad;jj < pad+1;jj++) {
          for(int ii  = -pad;ii < pad+1;ii++) {
            if( i+ii<0 or j+jj<0 or i+ii>=w or j+jj>=h) {
              r[count++] = 999;
            }else{
              r[count++] = img.at<cv::Vec3b>(i+ii,j+jj)[ch];
            }
          }
        }
        sort(r,r+fil_size*fil_size);
        out.at<cv::Vec3b>(i,j)[ch] = r[int(floor(count/2))+1];
      }
    }
  }
  return out;
}

cv::Mat answer10(cv::Mat img) {
  return Median(img);
}

int main(int argc, const char* argv[]){
  cv::Mat img = cv::imread("../Question_01_10/imori_noise.jpg",cv::IMREAD_COLOR);
  cv::Mat out = answer10(img);
  cv::imshow("sample", out);
  cv::waitKey(0);
  cv::destroyAllWindows(); 

  return 0;
}