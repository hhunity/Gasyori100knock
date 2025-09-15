// MarkerDetectSingle.hpp
// MarkerDetectWithOtsuSampling.hpp
#include <opencv2/opencv.hpp>
#include <cstdint>
#include <chrono>
#include <cstdio>

// ---- パラメータ --------------------------------------------------
struct RunParams {
    // 連続長条件（相対行）
    int minFirstWhite  = 100;
    int minBlack       = 950;
    int maxBlack       = 1050;
    int minSecondWhite = 100;

    // ヒステリシスしきい（0なら Otsu 自動）
    int thrWhiteHigh   = 185;   // 以上で白（0=自動）
    int thrBlackLow    = 120;   // 以下で黒（0=自動）
    int otsuMargin     = 30;    // thrWhiteHigh = otsu + margin

    // ★ Otsu 用ヒストグラム作成のサンプリング間隔
    //   1=全画素, 2=1つおき, 3=2つおき...
    int histStepX      = 1;     // 列方向ステップ
    int histStepY      = 1;     // 行方向ステップ

    // 穴あき許容（FSM 側）
    int blackHoleBudget  = 2;   // 黒区間中に許容する白行
    int whiteSpikeBudget = 2;   // 白区間中に許容する黒行

    // ROI（列方向）
    int roiX = 0;
    int roiW = -1;              // -1 で全幅

    // デバッグ
    int debug = 0;              // 0=OFF, 1=ON（stderr にログ）
};

// ---- 結果 --------------------------------------------------------
struct RunResultRel {
    bool      found      = false;
    int       blackFirst = -1;
    int       blackStart = -1;
    int       blackEnd   = -1;
    long long elapsed_us = 0;
    int       usedThrWhiteHigh = -1;
    int       usedThrBlackLow  = -1;
};

// ---- Otsu 閾値（サンプリング対応・コピーなし） -------------------
inline int computeOtsuThresholdSampled(const cv::Mat& view, int stepX, int stepY, uint64_t* outSampleCount = nullptr)
{
    CV_Assert(view.type() == CV_8UC1);
    if (stepX <= 0) stepX = 1;
    if (stepY <= 0) stepY = 1;

    uint64_t hist[256] = {0};
    uint64_t total = 0;

    for (int y = 0; y < view.rows; y += stepY) {
        const uint8_t* p = view.ptr<uint8_t>(y);
        for (int x = 0; x < view.cols; x += stepX) {
            ++hist[p[x]];
            ++total;
        }
    }
    if (outSampleCount) *outSampleCount = total;

    if (total == 0) return 127; // フォールバック

    // 全画素の重み付き平均（サンプルに対する）
    double sum = 0.0;
    for (int i = 0; i < 256; ++i) sum += (double)i * (double)hist[i];

    double sumB = 0.0;
    uint64_t wB = 0;
    double varMax = -1.0;
    int threshold = 0;

    for (int t = 0; t < 256; ++t) {
        wB += hist[t];
        if (wB == 0) continue;
        uint64_t wF = total - wB;
        if (wF == 0) break;

        sumB += (double)t * (double)hist[t];
        double mB = sumB / wB;
        double mF = (sum - sumB) / wF;

        double varBetween = (double)wB * (double)wF * (mB - mF) * (mB - mF);
        if (varBetween > varMax) { varMax = varBetween; threshold = t; }
    }
    return threshold;
}

// ---- 本体：1メソッドで完結（ヒステリシス＋穴あき許容＋Otsu自動） ---
inline RunResultRel processAndDetectWithHoles(const uint8_t* data, int W, int H, ptrdiff_t stride,
                                              RunParams p)
{
    RunResultRel out{};
    const auto t0 = std::chrono::high_resolution_clock::now();

    if (!data || W <= 0 || H <= 0) return out;
    if (stride <= 0) stride = W;

    // ROI
    cv::Mat m(H, W, CV_8UC1, const_cast<uint8_t*>(data), (size_t)stride);
    int x0 = (p.roiX < 0) ? 0 : (p.roiX > W ? W : p.roiX);
    int ww = (p.roiW > 0) ? p.roiW : W;
    if (ww > W - x0) ww = W - x0;
    if (ww <= 0) return out;

    cv::Mat view = m(cv::Rect(x0, 0, ww, H));

    // ---- Otsu しきい自動（必要時のみ・サンプリング対応） ----
    if (p.thrWhiteHigh == 0 || p.thrBlackLow == 0) {
        uint64_t sampled = 0;
        int otsuT = computeOtsuThresholdSampled(view, p.histStepX, p.histStepY, &sampled);
        if (p.thrBlackLow  == 0) p.thrBlackLow  = otsuT;
        if (p.thrWhiteHigh == 0) p.thrWhiteHigh = otsuT + p.otsuMargin;
        if (p.debug) {
            std::fprintf(stderr,
                "[DEBUG] Otsu(sampling X=%d,Y=%d) samples=%llu  T=%d  -> thrB=%d thrW=%d\n",
                p.histStepX, p.histStepY, (unsigned long long)sampled,
                otsuT, p.thrBlackLow, p.thrWhiteHigh);
        }
    }
    
    out.usedThrBlackLow  = p.thrBlackLow;
    out.usedThrWhiteHigh = p.thrWhiteHigh;

    // 行平均（0..255, float）
    cv::Mat rowMeanF; // H×1, CV_32F
    cv::reduce(view, rowMeanF, 1, cv::REDUCE_AVG, CV_32F);

    // FSM（穴あき許容）
    enum Phase { P_FirstWhite, P_Black, P_SecondWhite };
    Phase phase = P_FirstWhite;
    int lastBW = 0;   // +1=白, -1=黒, 0=未定
    int run = 0;      // 現フェーズの連続長
    int hole = 0;     // 黒フェーズでの白穴 使用数
    int spike = 0;    // 白フェーズでの黒スパイク 使用数
    int blackStart = -1, blackEnd = -1, blackFirst = -1;

    if (p.debug) {
        std::fprintf(stderr,
            "[DEBUG] start: W=%d H=%d ROI=(x=%d,w=%d) thrW>=%d thrB<=%d "
            "hole=%d spike=%d  req:W%d-B[%d..%d]-W%d\n",
            W, H, x0, ww, p.thrWhiteHigh, p.thrBlackLow,
            p.blackHoleBudget, p.whiteSpikeBudget,
            p.minFirstWhite, p.minBlack, p.maxBlack, p.minSecondWhite
        );
    }

    for (int y = 0; y < H; ++y) {
        float v = rowMeanF.at<float>(y);

        int bw = lastBW;
        if (v >= p.thrWhiteHigh) bw = +1;
        else if (v <= p.thrBlackLow) bw = -1;

        bool isBlack = (bw == -1);
        bool isWhite = !isBlack;
        if (p.debug) {
            char bwc = (bw==+1?'W':(bw==-1?'B':'?'));
            std::fprintf(stderr, "[DEBUG] y=%d v=%.1f bw=%c phase=%d run=%d hole=%d spike=%d\n",
                         y, v, bwc, (int)phase, run, hole, spike);
        }
        lastBW = bw;

        switch (phase) {
        case P_FirstWhite:
            if (isWhite) { ++run; spike = 0; }
            else {
                if (p.whiteSpikeBudget > 0 && spike < p.whiteSpikeBudget) { ++spike; }
                else { run = 0; spike = 0; }
            }
            if (run >= p.minFirstWhite) { phase = P_Black; run = 0; hole = 0; blackStart = -1; blackFirst = -1; }
            break;

        case P_Black:
            if (isBlack) {
                if (run == 0) { blackStart = y; blackFirst = y; }
                ++run; hole = 0;
                if (run > p.maxBlack) { blackStart = y; blackFirst = y; run = 1; hole = 0; }
            } else {
                if (p.blackHoleBudget > 0 && hole < p.blackHoleBudget) { ++hole; }
                else {
                    blackEnd = y - hole - 1;
                    if (run >= p.minBlack && run <= p.maxBlack) {
                        phase = P_SecondWhite; run = 1; spike = 0;
                    } else {
                        phase = P_FirstWhite; run = isWhite ? 1 : 0; spike = isWhite?0:1; blackStart = blackFirst = -1;
                    }
                }
            }
            break;

        case P_SecondWhite:
            if (isWhite) { ++run; spike = 0; }
            else {
                if (p.whiteSpikeBudget > 0 && spike < p.whiteSpikeBudget) { ++spike; }
                else { phase = P_FirstWhite; run = 0; spike = 0; blackStart = blackFirst = -1; }
            }
            if (run >= p.minSecondWhite) {
                out.found      = true;
                out.blackFirst = blackFirst;
                out.blackStart = blackStart;
                out.blackEnd   = blackEnd;
                const auto t1 = std::chrono::high_resolution_clock::now();
                out.elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
                return out;
            }
            break;
        }
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    out.elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    return out;
}

int main() {
    int W = 32, H = 512*4;
    cv::Mat img(H, W, CV_8UC1, cv::Scalar(255)); // 全白

    for (int y = 300; y < 1300; ++y) {
        img.row(y).setTo(0);
    }
    
    img.row(300).setTo(0);    //
    img.row(800).setTo(255);  //
    img.row(1600).setTo(0);   //

    RunParams prm; // デフォルト {100,950,1050,100,185,120,0,-1}
    prm.minFirstWhite  = 300;
    prm.minBlack       = 980;
    prm.maxBlack       = 1020;
    prm.minSecondWhite = 300;
    prm.thrWhiteHigh   = 180;  // 白判定しきい
    prm.thrBlackLow    = 50;  // 黒判定しきい
    prm.histStepX      = 4;         // 例: 横は4ピクセルおきにサンプル
    prm.histStepY      = 2;         // 例: 縦は2行おきにサンプル
    prm.roiX = 0;
    prm.roiW = -1;
    prm.blackHoleBudget   = 10;  // 黒区間中に許容する「白」行数（0で無効）
    prm.whiteSpikeBudget  = 10;  // 白区間中に許容する「黒」行数（0で無効）
    prm.debug = 0;

    RunResultRel r;
    long long time_sum = 0;
    int count_max = 100;
    
    for(int c = 0;c < count_max;c++) {
        r = processAndDetectWithHoles(img.data, img.cols, img.rows, img.step, prm);
        time_sum += r.elapsed_us;
    }

    if (r.found) {
        std::cout << "hit: first=" << r.blackFirst
                << " start=" << r.blackStart
                << " end="   << r.blackEnd
                << " (" << time_sum/count_max << " us)\n";
    } else {
        std::cout << "not found (" << time_sum/count_max  << " us)\n";
    }
}
