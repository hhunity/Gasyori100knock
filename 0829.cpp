// OpenCV CUDA sample - BufferPool + Streams + Tiled pipeline
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafft.hpp>   // OpenCV 4.x: <opencv2/cudaarithm.hpp> 内の dft でもOKな版あり
#include <vector>
#include <array>
#include <iostream>

using namespace cv;

struct Tile {
    Rect roi;
};

// ユーティリティ：16タイル分割（4x4）
static std::vector<Tile> makeTiles(Size full, int gridX = 4, int gridY = 4) {
    std::vector<Tile> tiles;
    int w = full.width / gridX;
    int h = full.height / gridY;
    for (int gy = 0; gy < gridY; ++gy) {
        for (int gx = 0; gx < gridX; ++gx) {
            Rect r(gx * w, gy * h, (gx == gridX-1 ? full.width - gx*w : w),
                                  (gy == gridY-1 ? full.height - gy*h : h));
            tiles.push_back({ r });
        }
    }
    return tiles;
}

int main() {
    // ---- 入力（CPU側）。実際はカメラ/ファイルから取得して CV_32F 化する想定 ----
    Mat hostA = imread("frameA.png", IMREAD_GRAYSCALE);
    Mat hostB = imread("frameB.png", IMREAD_GRAYSCALE);
    if (hostA.empty() || hostB.empty()) { std::cerr << "image missing\n"; return 1; }
    hostA.convertTo(hostA, CV_32F, 1.0/255.0);
    hostB.convertTo(hostB, CV_32F, 1.0/255.0);

    const Size fullSz = hostA.size();
    CV_Assert(hostB.size() == fullSz && hostA.type() == CV_32F && hostB.type() == CV_32F);

    // ---- ピン留め（転送高速化） ----
    cv::cuda::HostMem hA(hostA, cv::cuda::HostMem::PAGE_LOCKED);
    cv::cuda::HostMem hB(hostB, cv::cuda::HostMem::PAGE_LOCKED);

    // ---- GPU側フルフレーム（前確保・再利用） ----
    cv::cuda::GpuMat gA(fullSz, CV_32F);
    cv::cuda::GpuMat gB(fullSz, CV_32F);

    // ---- タイル分割情報 ----
    auto tiles = makeTiles(fullSz, 4, 4);
    constexpr int NSTREAMS = 4;                      // 4ストリームで16タイルを回す例
    std::array<cv::cuda::Stream, NSTREAMS> streams;

    // ---- 各ストリームに紐付いた BufferPool を取得 ----
    std::array<cv::Ptr<cv::cuda::BufferPool>, NSTREAMS> pools;
    for (int i = 0; i < NSTREAMS; ++i) {
        pools[i] = cv::cuda::getBufferPool(streams[i]); // ★ポイント：ここでプール紐付け
    }

    // ---- フィルタなどの“状態を持つオブジェクト”は使い回す ----
    // 例：Sobel（x,y 両方向）。出力は CV_32F 固定。
    auto sobelX = cv::cuda::createSobelFilter(CV_32F, CV_32F, 1, 0, 3);
    auto sobelY = cv::cuda::createSobelFilter(CV_32F, CV_32F, 0, 1, 3);

    // ---- 回転用の行列（例：-2.5度） ----
    const double angle_deg = -2.5;
    const Point2f center(fullSz.width*0.5f, fullSz.height*0.5f);
    Mat M = getRotationMatrix2D(center, angle_deg, 1.0);

    // ---- アップロード（非同期） ----
    gA.upload(hA, streams[0]);
    gB.upload(hB, streams[1]);

    // ---- 出力（例：各タイルの位相相関ピーク/オフセット）----
    std::vector<Point2f> peakOffsets(tiles.size(), Point2f(0,0));
    std::vector<double>    peakValues(tiles.size(), 0.0);

    // ---- タイル処理ループ ----
    for (size_t t = 0; t < tiles.size(); ++t) {
        int si = int(t % NSTREAMS);
        auto& s  = streams[si];
        auto& bp = pools[si];
        const Rect roi = tiles[t].roi;

        // ROI（フルフレームのビューなので追加割り当てなし）
        cv::cuda::GpuMat A = gA(roi);
        cv::cuda::GpuMat B = gB(roi);

        // ---- 1) 回転（B側だけ回転して A と照合する例）----
        //    一時領域はプールから借りる（Size/Type を固定にしてヒット率↑）
        cv::cuda::GpuMat B_rot = bp->getBuffer(roi.size(), CV_32F);
        // warpAffine の dst を前確保済み（上で getBuffer）にして内部再確保を防止
        cv::cuda::warpAffine(B, B_rot, M, roi.size(),
                             cv::INTER_LINEAR, cv::BORDER_CONSTANT, Scalar(0), s);

        // ---- 2) Sobel（オブジェクトの apply で内部ワーク再利用）----
        cv::cuda::GpuMat Ax = bp->getBuffer(roi.size(), CV_32F);
        cv::cuda::GpuMat Ay = bp->getBuffer(roi.size(), CV_32F);
        cv::cuda::GpuMat Bx = bp->getBuffer(roi.size(), CV_32F);
        cv::cuda::GpuMat By = bp->getBuffer(roi.size(), CV_32F);

        sobelX->apply(A,     Ax, s);
        sobelY->apply(A,     Ay, s);
        sobelX->apply(B_rot, Bx, s);
        sobelY->apply(B_rot, By, s);

        // 勾配強度（optional）：|grad| = sqrt(x^2 + y^2)
        cv::cuda::GpuMat Agrad = bp->getBuffer(roi.size(), CV_32F);
        cv::cuda::GpuMat Bgrad = bp->getBuffer(roi.size(), CV_32F);
        cv::cuda::magnitude(Ax, Ay, Agrad, s);
        cv::cuda::magnitude(Bx, By, Bgrad, s);

        // ---- 3) FFT（複素数 2チャンネル：CV_32FC2）----
        //     dft(src,dst,flags,nonzeroRows,stream) の形（OpenCV 4.x）
        Size dsz = roi.size();
        cv::cuda::GpuMat FA = bp->getBuffer(dsz, CV_32FC2);
        cv::cuda::GpuMat FB = bp->getBuffer(dsz, CV_32FC2);

        cv::cuda::dft(Agrad, FA, dsz, 0, s);
        cv::cuda::dft(Bgrad, FB, dsz, 0, s);

        // ---- 4) 相互パワースペクトル：FA * conj(FB) / |FA * conj(FB)| ----
        //     要素ごと複素演算（2ch）。OpenCV CUDA では専用関数が無いので自前カーネルが理想だが、
        //     手短に “FB を共役化 → 複素乗算 → 正規化” を arithm のブロックで実装。
        //     ここでは簡略化のため、FB を共役化してから複素乗算・正規化を行う。
        // 共役：imag を -1 倍
        {
            std::vector<cv::cuda::GpuMat> ch(2);
            cv::cuda::split(FB, ch, s);                 // ch[0]=Re, ch[1]=Im
            cv::cuda::multiply(ch[1], cv::Scalar(-1.0f), ch[1], 1.0, -1, s);
            cv::cuda::merge(ch, FB, s);
        }

        cv::cuda::GpuMat Fcross = bp->getBuffer(dsz, CV_32FC2);
        cv::cuda::mulSpectrums(FA, FB, Fcross, 0/* flags=DFT_ROWS off */, true/* conjB already done? -> keep true */, s);
        // 振幅で正規化（|Fcross| で割る）
        {
            std::vector<cv::cuda::GpuMat> ch(2);
            cv::cuda::split(Fcross, ch, s);
            cv::cuda::GpuMat mag = bp->getBuffer(dsz, CV_32F);
            cv::cuda::magnitude(ch[0], ch[1], mag, s);
            // avoid divide-by-zero: max(mag, eps)
            cv::cuda::GpuMat denom = bp->getBuffer(dsz, CV_32F);
            cv::cuda::max(mag, 1e-12, denom, s);
            ch[0].convertTo(ch[0], CV_32F, 1.0); // 明示的に型を意識（既に32Fのはず）
            ch[1].convertTo(ch[1], CV_32F, 1.0);
            cv::cuda::divide(ch[0], denom, ch[0], 1.0, -1, s);
            cv::cuda::divide(ch[1], denom, ch[1], 1.0, -1, s);
            cv::cuda::merge(ch, Fcross, s);
        }

        // ---- 5) 逆FFT → 相関面（実数）----
        cv::cuda::GpuMat corr = bp->getBuffer(dsz, CV_32F);
        cv::cuda::dft(Fcross, corr, dsz, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE, s);

        // ---- 6) ピーク検出（GPU上で minMaxLoc）----
        double minVal, maxVal; Point minLoc, maxLoc;
        cv::cuda::minMaxLoc(corr, &minVal, &maxVal, &minLoc, &maxLoc, cv::cuda::GpuMat(), s);

        // ストリーム完了を待ってから結果を読む（loc はホスト側で読む必要がある）
        s.waitForCompletion();

        // 亜ピクセル化などは省略。単純にピーク座標を採用。
        // wrap-around を考慮して [-W/2, W/2) に正規化（相互相関の定番）
        Point2f shift(maxLoc.x, maxLoc.y);
        if (shift.x > dsz.width/2)  shift.x -= dsz.width;
        if (shift.y > dsz.height/2) shift.y -= dsz.height;

        peakOffsets[t] = shift;
        peakValues[t]  = maxVal;
        // 以降、借りたテンポラリはスコープ終了で自動的にプールへ返却
    }

    // すべてのストリーム完了待ち（安全のため）
    for (auto& s : streams) s.waitForCompletion();

    // ---- 結果の表示（例）----
    for (size_t i = 0; i < tiles.size(); ++i) {
        std::cout << "tile " << i
                  << " peak=(" << peakOffsets[i].x << "," << peakOffsets[i].y
                  << ") val=" << peakValues[i] << "\n";
    }
    return 0;
}