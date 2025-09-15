// MarkerDetect.hpp
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdint>
#include <algorithm>

// -------------------------------
// 行平均→ヒステリシスで isBlack ベクトルを作る
// -------------------------------
class BlockToIsBlack {
public:
    int thrWhiteHigh = 185;  // この値以上なら白
    int thrBlackLow  = 120;  // この値以下なら黒
    int roiX = 0, roiW = -1; // ROI（全幅: x=0, w=-1）

    // ブロック (8UC1, W×H) を処理して isBlack[0..H-1] を返す
    std::vector<bool> process(const uint8_t* data, int W, int H, ptrdiff_t stride) {
        std::vector<bool> result;
        if (!data || W <= 0 || H <= 0) return result;
        if (stride <= 0) stride = W;

        cv::Mat m(H, W, CV_8UC1, const_cast<uint8_t*>(data), (size_t)stride);

        int x0 = (roiX < 0) ? 0 : (roiX > W ? W : roiX);
        int ww = (roiW > 0) ? roiW : W;
        if (ww > W - x0) ww = W - x0;
        if (ww <= 0) return result;

        cv::Mat view = m(cv::Rect(x0, 0, ww, H));

        // 行平均 (float)
        cv::Mat rowMeanF;
        cv::reduce(view, rowMeanF, 1, cv::REDUCE_AVG, CV_32F);

        result.resize(H);
        int lastBW = 0; // +1=白, -1=黒, 0=未定

        for (int y = 0; y < H; ++y) {
            float v = rowMeanF.at<float>(y);

            int bw = lastBW;
            if (v >= thrWhiteHigh) bw = +1;     // 白
            else if (v <= thrBlackLow) bw = -1; // 黒
            // 中間は前の状態を維持

            result[y] = (bw == -1); // true=黒
            lastBW = bw;
        }
        return result;
    }
};

// -------------------------------
// 相対行インデックスで黒帯検出
// -------------------------------
struct RunParams {
    int minFirstWhite  = 100;
    int minBlack       = 950;
    int maxBlack       = 1050;
    int minSecondWhite = 100;
};

struct RunResultRel {
    bool found      = false;
    int  blackFirst = -1;  // 最初の黒行
    int  blackStart = -1;  // 黒ラン開始
    int  blackEnd   = -1;  // 黒ラン終了 (inclusive)
};

// isBlack[0..N-1] を走査して、白→黒→白のランを探す
inline RunResultRel detectRunRelative(const std::vector<bool>& isBlack, const RunParams& p)
{
    RunResultRel out;
    enum Phase { P_FirstWhite, P_Black, P_SecondWhite } phase = P_FirstWhite;
    int run = 0;
    int blackStart = -1, blackEnd = -1, blackFirst = -1;

    const int N = (int)isBlack.size();
    for (int y = 0; y < N; ++y) {
        const bool isW = !isBlack[y];
        switch (phase) {
        case P_FirstWhite:
            run = isW ? (run + 1) : 0;
            if (run >= p.minFirstWhite) { phase = P_Black; run = 0; }
            break;

        case P_Black:
            if (!isW) {
                if (run == 0) { blackStart = y; blackFirst = y; }
                ++run;
                if (run > p.maxBlack) { // 上限超過→この行からリスタート
                    blackStart = y;
                    blackFirst = y;
                    run = 1;
                }
            } else {
                if (run >= p.minBlack && run <= p.maxBlack) {
                    blackEnd = y - 1;
                    phase = P_SecondWhite; run = 0;
                } else {
                    phase = P_FirstWhite; run = isW ? 1 : 0;
                    blackStart = blackFirst = -1;
                }
            }
            break;

        case P_SecondWhite:
            run = isW ? (run + 1) : 0;
            if (run >= p.minSecondWhite) {
                out.found      = true;
                out.blackFirst = blackFirst;
                out.blackStart = blackStart;
                out.blackEnd   = blackEnd;
                return out; // 1件見つけたら即返す
            }
            break;
        }
    }
    return out; // 見つからず
}
