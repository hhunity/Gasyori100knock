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
    void   operator+= (rgb a) {b+=a.b;g+=a.g;r+=a.r;}
    void   operator/= (int c) {b/=c  ;g/=c  ;r/=c  ;}
    bool   operator!= (rgb a) const {return b==a.b and g==a.b and r==a.r;}

    double calc_distance(rgb a) {
        return  sqrt( pow(r-a.r,2)+pow(g-a.g,2)+pow(b-a.b,2) );
    }
};

rgb get_rgb(const uchar *pos){
    return  rgb(pos[2],pos[1],pos[0]);
}

cv::Mat kmeans_quantam_color(cv::Mat img,int k = 5)
{
    mt19937 gen(0);
    int img_size = img.rows*img.cols;
    uniform_int_distribution<int> dist(0,img_size-1);
    auto sampling_rgb = std::make_unique<rgb[]>(k);
    auto img_index    = std::make_unique<uchar[]>(img_size);
    auto average      = std::make_unique<rgb[]>(k);
    int ch_num=img.channels();
    int compare_count = 1;

    for(int i = 0;i< k;i++ ) {
        uchar *pos = img.data+dist(gen)*ch_num;
        sampling_rgb[i] = get_rgb(pos);
        // sampling_rgb[i].show();
    }

    while(compare_count>0) {
        compare_count = 0;
        for(int i = 0;i<img_size; i++) {
            uchar *pos = img.data+i*ch_num;
            rgb value  = get_rgb(pos);
            double distance;
            double min_distace = img_size*4;
            int min_index = 0;
            for(int j = 0 ; j< k ;j++) {
                distance = value.calc_distance(sampling_rgb[j]);
                if(distance <= min_distace ){
                    min_distace = distance;
                    min_index   = j;
                }
            }
            img_index[i] = min_index;
        }
        for(int i = 0;i<img_size; i++) {
            uchar *pos = img.data+i*ch_num;
            rgb value  = get_rgb(pos);
            average[img_index[i]] += value;
        }

        for(int i = 0;i< k ;i++ ) {
            // average[i].show();
            int sum = 0;
            for(int j=0;j<img_size;j++) {
                if(img_index[j]==i) {
                    sum+=1;
                }
            }
            
            average[i] /= sum;
            if( average[i] != sampling_rgb[i]) {
                compare_count++;
            }
            sampling_rgb[i] = average[i];
            // sampling_rgb[i].show();
        }
    }

    for(int i = 0;i<img_size; i++) {
        uchar *pos = img.data+i*ch_num;
        pos[0] = sampling_rgb[img_index[i]].b;
        pos[1] = sampling_rgb[img_index[i]].g;
        pos[2] = sampling_rgb[img_index[i]].r;
    }

    return img;
}

int main() {
	auto start = std::chrono::high_resolution_clock::now();

    vector<vector<double>> train_x = {{0, 0}, {0, 1},{1, 0},{1, 1}};
    vector<double> train_t = {0, 1, 1, 0};

    cv::Mat img = cv::imread("../Question_91_100/imori.jpg", cv::IMREAD_COLOR);
    
    img = kmeans_quantam_color(img,10);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "time:" << duration.count()/1000 <<  std::endl;

    cv::imshow("img",img);
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    return 0;
}