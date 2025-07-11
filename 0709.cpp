
using System;
using BitMiracle.LibTiff.Classic;

public static class TiffWriter
{
    public static void SaveGrayscaleTiff(string outputPath, byte[] buffer, int width, int height)
    {
        if (buffer.Length != width * height)
            throw new ArgumentException("バッファサイズが画像サイズと一致しません");

        using (Tiff output = Tiff.Open(outputPath, "w"))
        {
            output.SetField(TiffTag.IMAGEWIDTH, width);
            output.SetField(TiffTag.IMAGELENGTH, height);
            output.SetField(TiffTag.SAMPLESPERPIXEL, 1);
            output.SetField(TiffTag.BITSPERSAMPLE, 8);
            output.SetField(TiffTag.ROWSPERSTRIP, height); // 1ストリップ全体
            output.SetField(TiffTag.COMPRESSION, Compression.LZW); // 無圧縮なら Compression.NONE
            output.SetField(TiffTag.PHOTOMETRIC, Photometric.MINISBLACK); // 0=黒, 255=白
            output.SetField(TiffTag.ORIENTATION, Orientation.TOPLEFT);
            output.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG);

            int stride = width; // 1ピクセル=1バイト

            for (int row = 0; row < height; row++)
            {
                output.WriteScanline(buffer, row * stride, row, 0);
            }

            output.WriteDirectory();
        }
    }
}



private void pictureBox1_MouseMove(object sender, MouseEventArgs e)
{
    if (pictureBox1.Image == null)
        return;

    var img = pictureBox1.Image;

    // ピクチャーボックスのサイズ
    int pbW = pictureBox1.Width;
    int pbH = pictureBox1.Height;

    // 画像のサイズ
    int imgW = img.Width;
    int imgH = img.Height;

    // スケールの計算
    float ratioX = (float)pbW / imgW;
    float ratioY = (float)pbH / imgH;
    float ratio = Math.Min(ratioX, ratioY); // Zoomは縦横比を保つ

    // 実際の表示サイズ
    int displayW = (int)(imgW * ratio);
    int displayH = (int)(imgH * ratio);

    // 中央寄せオフセット
    int offsetX = (pbW - displayW) / 2;
    int offsetY = (pbH - displayH) / 2;

    // マウス位置（PictureBox内）→ 画像位置
    int x = (int)((e.X - offsetX) / ratio);
    int y = (int)((e.Y - offsetY) / ratio);

    // 範囲外チェック
    if (x >= 0 && x < imgW && y >= 0 && y < imgH)
    {
        this.Text = $"Image coordinates: ({x}, {y})";
    }
    else
    {
        this.Text = "Outside image area";
    }
}


#include <opencv2/opencv.hpp>
#include <iostream>

void saveDebugImages(const cv::Mat& img1, const cv::Mat& img2,
                     const cv::Mat& dft1, const cv::Mat& dft2,
                     const cv::Mat& numerator, const cv::Mat& corr)
{
    // 振幅スペクトルの表示用画像に変換
    auto computeLogMagnitude = [](const cv::Mat& complex) -> cv::Mat {
        std::vector<cv::Mat> planes;
        cv::split(complex, planes);
        cv::Mat mag;
        cv::magnitude(planes[0], planes[1], mag);
        mag += cv::Scalar::all(1); // log(0) 対策
        cv::log(mag, mag);
        cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX);
        mag.convertTo(mag, CV_8UC1);
        return mag;
    };

    // 相関マップの可視化
    cv::Mat corr_vis;
    cv::normalize(corr, corr_vis, 0, 255, cv::NORM_MINMAX);
    corr_vis.convertTo(corr_vis, CV_8UC1);
    cv::applyColorMap(corr_vis, corr_vis, cv::COLORMAP_HOT);

    // 各画像を保存
    cv::imwrite("debug_input1.png", img1);
    cv::imwrite("debug_input2.png", img2);
    cv::imwrite("debug_fft1.png", computeLogMagnitude(dft1));
    cv::imwrite("debug_fft2.png", computeLogMagnitude(dft2));
    cv::imwrite("debug_cross_power.png", computeLogMagnitude(numerator));
    cv::imwrite("debug_correlation_map.png", corr_vis);

    std::cout << "Debug images saved.\n";
}


#include <opencv2/opencv.hpp>
#include <cmath>

cv::Mat applyBandpassFilter(const cv::Mat& dft, double low = 10.0, double high = 100.0)
{
    CV_Assert(dft.type() == CV_32FC2); // 複素数（real, imag）2チャンネル

    int h = dft.rows;
    int w = dft.cols;
    int cx = w / 2;
    int cy = h / 2;

    // 出力をコピー
    cv::Mat filtered = dft.clone();

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            double dy = y - cy;
            double dx = x - cx;
            double dist = std::sqrt(dx * dx + dy * dy);

            if (dist <= low || dist >= high)
            {
                // バンドの外はゼロにする（実部・虚部両方）
                filtered.at<cv::Vec2f>(y, x) = cv::Vec2f(0.0f, 0.0f);
            }
        }
    }

    return filtered;
}


cv::Mat img; // CV_32FC1 グレースケール画像
cv::Mat dft;
cv::dft(img, dft, cv::DFT_COMPLEX_OUTPUT);

cv::Mat filtered_dft = applyBandpassFilter(dft, 20.0, 120.0);

// フィルター後に逆変換（必要なら）
cv::Mat filtered_img;
cv::dft(filtered_dft, filtered_img, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);




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



