
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



