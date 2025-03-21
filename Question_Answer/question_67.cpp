#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cblas.h>  // BLASライブラリを使用
#include <iomanip>
#include <memory>

using namespace std;

void debug(string str="") {
    cout << "debug:" << str << endl;
}

struct RGB {
    long b;
    long g;
    long r;
    RGB(){}
    RGB(const uchar *pos):b(pos[2]),g(pos[1]),r(pos[0]){}
    RGB(int r,int g,int b) : b(b),g(g),r(r){}
    void show(){
        cout << b << "," << g << "," << g << endl;
    }
    RGB&   operator+= (RGB a) {b+=a.b;g+=a.g;r+=a.r;return *this;}
    RGB&   operator/= (int c) {b/=c  ;g/=c  ;r/=c  ;return *this;}
    bool   operator!= (RGB a) const {return b==a.b and g==a.b and r==a.r;}
    uchar  get_y(){return 0.2126*r + 0.7152*g + 0.0722*b;}
    double calc_distance(RGB a) {
        return  sqrt( pow(r-a.r,2)+pow(g-a.g,2)+pow(b-a.b,2) );
    }
};

struct Posi {
    int x;
    int y;
    Posi(int x,int y ):x(x),y(y) {}
    void show(){
        cout << x <<","<<y << endl;
    }
};

struct Rect {
    Posi p0;
    Posi p1;
    Rect(int x1,int y1,int x2,int y2) : p0(x1,y1),p1(x2,y2) {}
    cv::Rect get_rect(){return cv::Rect(p0.x,p0.y,p1.x-p0.x,p1.y-p0.y);}
    int get_width(){  return p1.x-p0.x;}
    int get_height(){ return p1.y-p0.y;}
    void show(){
        p0.show();
        p1.show();
    }
};

double intersection_over_union(Rect gt,Rect rec) {
    int area_gt = gt.get_width()*gt.get_height();
    int area_a  = rec.get_width()*rec.get_height();

    int x0 = max(gt.p0.x,rec.p0.x);
    int x1 = min(gt.p1.x,rec.p1.x);
    int y0 = max(gt.p0.y,rec.p0.y);
    int y1 = min(gt.p1.y,rec.p1.y);

    double rol = (x1-x0)*(y1-y0);

    return (rol)/(area_gt+area_a-rol);

}
void get_y(cv::Mat& img,cv::Mat& gray){
    int H = img.rows;
    int W = img.cols;
    int color_num = img.channels();
    uchar *pos = img.data;

    for(int i = 0;i < W*H;i++) {
        RGB rgb(pos+i*color_num);
        gray.data[i] = rgb.get_y();
    }

}

void get_hog_mag_ang(cv::Mat& gray,unique_ptr<uchar[]>& img_mag,unique_ptr<uchar[]>& img_ang) {
    int H = gray.rows;
    int W = gray.cols;
    uchar *pos = gray.data;

    for(int y = 0;y< H ;y++) {
        for(int x = 0;x <W;x++ ) {
            int xx = min(x+1,W-1);
            int yy = min(y+1,H-1);
            int gx = pos[y *W+xx]- pos[y*W+x];
            int gy = pos[yy*W+x] - pos[y*W+x];

            img_mag[y*W+x] = sqrt(pow(gx,2)+pow(gy,2));
            
            double ang = (gx!=0) ? atan(gy/gx)*180.0/M_PI : 0;
            ang+=90;
            img_ang[y*W+x] = floor(ang/20);
        }
    }

    return;
}

void get_hog_ang_hist(unique_ptr<uchar[]>& img_mag,unique_ptr<uchar[]>& img_ang,int H,int W,unique_ptr<uchar[]>& ang_hist) {

    for(int y = 0;y< H ;y++) {
        for(int x = 0;x <W;x++ ) {
            int ang = img_ang[y*W+x];
            int xx = floor(x/8);
            int yy = floor(y/8);
            ang_hist[yy*W/8+xx + ang*(H/8*W/8)] += img_mag[y*W+x];
        }
    }
}

int main() {
	auto start = std::chrono::high_resolution_clock::now();

    cv::Mat img = cv::imread("../Question_91_100/imori.jpg", cv::IMREAD_COLOR);
    cv::Mat gray= cv::Mat(img.rows,img.cols,CV_8UC1);
    int   ang_hist_size = (img.cols/8*img.rows/8)*9;
    auto  img_mag  = std::make_unique<uchar[]>(img.cols*img.rows);
    auto  img_ang  = std::make_unique<uchar[]>(img.cols*img.rows);
    auto  ang_hist = std::make_unique<uchar[]>(ang_hist_size);

    fill(ang_hist.get(),ang_hist.get()+ang_hist_size,0);

    get_y(img,gray);
    get_hog_mag_ang(gray,img_mag,img_ang);
    get_hog_ang_hist(img_mag,img_ang,img.rows,img.cols,ang_hist);

    cv::Mat out = cv::Mat(gray.rows/8,img.cols/8,CV_8UC1,ang_hist.get());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "time:" << duration.count()/1000 <<  std::endl;

    cv::imshow("img",out);
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    return 0;
}