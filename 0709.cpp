
#include <opencv2/opencv.hpp>
#include <cmath>

cv::Mat maskCorrNearShift(const cv::Mat& corr, cv::Point2d prev_shift, double max_distance = 10.0)
{
    CV_Assert(corr.type() == CV_32FC1 || corr.type() == CV_64FC1);

    int h = corr.rows;
    int w = corr.cols;

    // wrap-around 対応の位置
    int cx = static_cast<int>(std::round(std::fmod(prev_shift.x + w, w)));
    int cy = static_cast<int>(std::round(std::fmod(prev_shift.y + h, h)));

    // 結果マスク初期化
    cv::Mat masked = cv::Mat::zeros(corr.size(), corr.type());

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            double dx = x - cx;
            double dy = y - cy;
            double dist = std::sqrt(dx * dx + dy * dy);

            if (dist <= max_distance)
            {
                if (corr.type() == CV_32FC1)
                    masked.at<float>(y, x) = corr.at<float>(y, x);
                else
                    masked.at<double>(y, x) = corr.at<double>(y, x);
            }
        }
    }

    return masked;
}

cv::Mat corr = ...; // 相関マップ（CV_32FC1 or CV_64FC1）
cv::Point2d last_shift(23.4, 45.8);

cv::Mat masked = maskCorrNearShift(corr, last_shift, 15.0);




#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

std::vector<double> removeOutliers(const std::vector<double>& data, double z_thresh = 2.5)
{
    std::vector<double> filtered;

    // 平均を計算
    double sum = 0.0;
    for (double val : data) sum += val;
    double mean = sum / data.size();

    // 標準偏差を計算
    double sq_sum = 0.0;
    for (double val : data) sq_sum += (val - mean) * (val - mean);
    double stddev = std::sqrt(sq_sum / data.size());

    // Zスコア判定
    for (double val : data)
    {
        double z = (val - mean) / (stddev + 1e-10); // ゼロ除算防止
        if (std::abs(z) < z_thresh)
            filtered.push_back(val);
    }

    return filtered;
}

std::vector<double> values = {1.0, 1.1, 1.2, 5.0, 1.3, 1.2}; // 5.0 は外れ値
auto filtered = removeOutliers(values, 2.0);



