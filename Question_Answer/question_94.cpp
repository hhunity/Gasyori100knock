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

struct rgb {
    long b;
    long g;
    long r;
    rgb(){}
    rgb(int r,int g,int b) : b(b),g(g),r(r){}
    void show(){
        cout << b << "," << g << "," << g << endl;
    }
    rgb&   operator+= (rgb a) {b+=a.b;g+=a.g;r+=a.r;return *this;}
    rgb&   operator/= (int c) {b/=c  ;g/=c  ;r/=c  ;return *this;}
    bool   operator!= (rgb a) const {return b==a.b and g==a.b and r==a.r;}

    double calc_distance(rgb a) {
        return  sqrt( pow(r-a.r,2)+pow(g-a.g,2)+pow(b-a.b,2) );
    }
};

rgb get_rgb(const uchar *pos){
    return  rgb(pos[2],pos[1],pos[0]);
}

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

cv::Mat get_mm_data(cv::Mat img,int N=200) {
    Rect gt{47,41,129,103};
    mt19937 gen(0);
    int img_size = img.rows*img.cols;
    int W=60,H=60;
    uniform_int_distribution<int> dist_x(0,img.cols-W-1);
    uniform_int_distribution<int> dist_y(0,img.rows-H-1);

    for(int i=0;i<N;i++) {
        int x0 = dist_x(gen);
        int y0 = dist_y(gen);
        Rect cut{x0,y0,x0+W,y0+H};
        double iou = intersection_over_union(gt,cut);
        cv::Scalar color{255,0,0};
        if(iou >=0.5) {
            color = cv::Scalar(0,0,255);
        }

        cv::rectangle(img,cut.get_rect(),color,1);

    }

    cv::rectangle(img,gt.get_rect(),cv::Scalar(0,255,0),1);

    return img;
}

int main() {
	auto start = std::chrono::high_resolution_clock::now();

    vector<vector<double>> train_x = {{0, 0}, {0, 1},{1, 0},{1, 1}};
    vector<double> train_t = {0, 1, 1, 0};

    cv::Mat img = cv::imread("../Question_91_100/imori_1.jpg", cv::IMREAD_COLOR);
    
    img = get_mm_data(img);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "time:" << duration.count()/1000 <<  std::endl;

    cv::imshow("img",img);
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    return 0;
}