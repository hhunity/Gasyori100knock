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
    int b;
    int g;
    int r;
    void show(){
        cout << b << "," << g << "," << g << endl;
    }
    double operator- (rgb a) const {return pow(r-a.r,2)+pow(g-a.g,2)+pow(b-a.b,2);}
};

rgb get_rgb(const uchar *pos){
    rgb value;

    value.b =  pos[0];
    value.g =  pos[1];
    value.r =  pos[2];

    return  value;
}

double calc_distance(const rgb &value1,const rgb &value2)
{
    return sqrt(value1 - value2);
}

cv::Mat kmeans_quantam_color(cv::Mat img,int k = 5)
{
    mt19937 gen(0);
    int img_size = img.rows*img.cols;
    uniform_int_distribution<int> dist(0,img_size-1);
    auto sampling_rgb = std::make_unique<rgb[]>(k);
    auto img_index    = std::make_unique<uchar[]>(img_size);
    int ch_num=img.channels();

    for(int i = 0;i< k;i++ ) {
        uchar *pos = img.data+dist(gen)*ch_num;
        sampling_rgb[i] = get_rgb(pos);
        // sampling_rgb[i].show();
    }

    for(int i = 0;i<img_size; i++) {
        uchar *pos = img.data+i*ch_num;
        rgb value  = get_rgb(pos);
        double distance;
        double min_distace = img_size*4;
        int min_index = 0;
        for(int j = 0 ; j< k ;j++) {
            distance = calc_distance(value,sampling_rgb[j]);
            if(distance <= min_distace ){
                min_distace = distance;
                min_index   = j;
            }
        }
        img_index[i] = min_index*50;
    }

    cv::Mat gray(img.cols,img.rows,CV_8UC1,static_cast<void*>(img_index.get()));
    cv::Mat img_clone = gray.clone();  // データを独立させる！
    return img_clone;
}

int main() {
	auto start = std::chrono::high_resolution_clock::now();

    vector<vector<double>> train_x = {{0, 0}, {0, 1},{1, 0},{1, 1}};
    vector<double> train_t = {0, 1, 1, 0};

    cv::Mat img = cv::imread("../Question_91_100/imori.jpg", cv::IMREAD_COLOR);
    
    img = kmeans_quantam_color(img);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "time:" << duration.count()/1000 <<  std::endl;

    cv::imshow("img",img);
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    return 0;
}