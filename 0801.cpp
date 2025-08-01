

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;

class Program
{
    [DllImport("winmm.dll")]
    private static extern uint timeBeginPeriod(uint uMilliseconds);
    [DllImport("winmm.dll")]
    private static extern uint timeEndPeriod(uint uMilliseconds);

    static void PreciseSleep(double targetMilliseconds)
    {
        timeBeginPeriod(1); // タイマ精度を1msに設定（必須）

        Stopwatch sw = Stopwatch.StartNew();

        // 長い時間はSleepで
        if (targetMilliseconds > 5)
            Thread.Sleep((int)(targetMilliseconds - 2)); // 少し手前で止める

        // 残りをbusy waitで精密に調整
        while (sw.Elapsed.TotalMilliseconds < targetMilliseconds) { }

        timeEndPeriod(1);
    }

    static void Main()
    {
        for (int i = 0; i < 5; i++)
        {
            Stopwatch sw = Stopwatch.StartNew();
            PreciseSleep(100.0);  // 100ms精度 ±0.5ms程度
            sw.Stop();
            Console.WriteLine($"Slept: {sw.Elapsed.TotalMilliseconds:F3} ms");
        }
    }
}


///cuda graph

#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafourier.hpp>

const int IMG_SIZE = 1024;
const int TILE_SIZE = 256;
const int TILE_COUNT = 16;

struct TileBuffers {
    cv::cuda::GpuMat tile;
    cv::cuda::GpuMat tile_f32;
    cv::cuda::GpuMat sobel_x, sobel_y;
    cv::cuda::GpuMat magnitude;
    cv::cuda::GpuMat fft_result;
};

void processWithCudaGraph(const std::vector<cv::Mat>& input_frames) {
    // ---- 固定領域確保 ----
    cv::cuda::GpuMat input_gpu(IMG_SIZE, IMG_SIZE, CV_8UC1);
    cv::cuda::GpuMat rotated_gpu;

    // 回転行列（45度）
    cv::Mat affine = cv::getRotationMatrix2D(cv::Point2f(IMG_SIZE / 2.0f, IMG_SIZE / 2.0f), 45.0, 1.0);

    // タイルごとのバッファ
    std::vector<TileBuffers> tiles(TILE_COUNT);
    for (auto& tb : tiles) {
        tb.tile = cv::cuda::GpuMat(TILE_SIZE, TILE_SIZE, CV_8UC1);
        tb.tile_f32.create(TILE_SIZE, TILE_SIZE, CV_32F);
        tb.sobel_x.create(TILE_SIZE, TILE_SIZE, CV_32F);
        tb.sobel_y.create(TILE_SIZE, TILE_SIZE, CV_32F);
        tb.magnitude.create(TILE_SIZE, TILE_SIZE, CV_32F);
        tb.fft_result.create(TILE_SIZE, TILE_SIZE, CV_32FC2);  // FFT結果は複素数
    }

    // フィルター（共通）
    auto sobelX = cv::cuda::createSobelFilter(CV_32F, -1, 1, 0);
    auto sobelY = cv::cuda::createSobelFilter(CV_32F, -1, 0, 1);

    // ストリームとグラフ
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    // ---- Graph キャプチャ開始 ----
    input_gpu.upload(input_frames[0], stream);  // 仮の1枚

    // 回転
    cv::cuda::warpAffine(input_gpu, rotated_gpu, affine, cv::Size(IMG_SIZE, IMG_SIZE), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(), stream);

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    // タイル処理
    int tile_idx = 0;
    for (int y = 0; y < IMG_SIZE; y += TILE_SIZE) {
        for (int x = 0; x < IMG_SIZE; x += TILE_SIZE) {
            auto& tb = tiles[tile_idx++];
            cv::cuda::GpuMat roi = rotated_gpu(cv::Rect(x, y, TILE_SIZE, TILE_SIZE));
            roi.copyTo(tb.tile, stream);  // tile = ROI

            // float変換 → Sobel → magnitude → FFT
            tb.tile.convertTo(tb.tile_f32, CV_32F, 1.0, 0.0, stream);
            sobelX->apply(tb.tile_f32, tb.sobel_x, stream);
            sobelY->apply(tb.tile_f32, tb.sobel_y, stream);
            cv::cuda::magnitude(tb.sobel_x, tb.sobel_y, tb.magnitude, stream);
            cv::cuda::dft(tb.magnitude, tb.fft_result, TILE_SIZE, stream);
        }
    }

    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

    // ---- 実行ループ（毎フレーム） ----
    for (const auto& frame : input_frames) {
        input_gpu.upload(frame, stream);  // 入力を更新（同じアドレス）

        // グラフで処理（回転 + 16タイルの sobel + fft）
        cudaGraphLaunch(graphExec, stream);
        cudaStreamSynchronize(stream);

        // fft_result に16個分の結果が格納済み（GPU上）
    }

    // 後始末
    cudaStreamDestroy(stream);
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(graphExec);
}




//////cuda graph

cudaStream_t stream;
cudaStreamCreate(&stream);

// 1. タイルを float32 に変換（CV_32F）
cv::cuda::GpuMat tile = rotated_gpu(cv::Rect(0, 0, 256, 256));
cv::cuda::GpuMat tile_float;
tile.convertTo(tile_float, CV_32F, 1.0, 0.0, stream);

// 2. 各バッファ（出力）用意
cv::cuda::GpuMat sobel_x, sobel_y, magnitude, fft_result;

// 3. フィルター作成（固定）
auto sobelX = cv::cuda::createSobelFilter(tile_float.type(), -1, 1, 0);
auto sobelY = cv::cuda::createSobelFilter(tile_float.type(), -1, 0, 1);

// 4. グラフ構築開始
cudaGraph_t graph;
cudaGraphExec_t instance;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// --- 以下の処理がグラフに記録される ---
sobelX->apply(tile_float, sobel_x, stream);
sobelY->apply(tile_float, sobel_y, stream);
cv::cuda::magnitude(sobel_x, sobel_y, magnitude, stream);
cv::cuda::dft(magnitude, fft_result, magnitude.size(), stream);
// --- ここまで ---

cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);

// 5. グラフ実行（高速！）
for (int i = 0; i < 1000; ++i) {
    cudaGraphLaunch(instance, stream);
}
// 待機
cudaStreamSynchronize(stream);



////roi 非同期処理

void process_full_async(const cv::Mat& input_cpu) {
    const int TILE_SIZE = 256;
    const int IMG_SIZE = 1024;
    const int NUM_TILES = 16;

    // ストリーム & バッファ準備
    std::vector<cv::cuda::Stream> streams(NUM_TILES);
    std::vector<cv::cuda::GpuMat> sobel_xs(NUM_TILES);
    std::vector<cv::cuda::GpuMat> sobel_ys(NUM_TILES);
    std::vector<cv::cuda::GpuMat> magnitudes(NUM_TILES);
    std::vector<cv::cuda::GpuMat> fft_results(NUM_TILES);

    // 1. GPUにアップロード
    cv::cuda::GpuMat input_gpu, rotated_gpu;
    input_gpu.upload(input_cpu);

    // 2. 回転（warpAffineは1回でOK）
    cv::Mat affine = cv::getRotationMatrix2D(cv::Point2f(IMG_SIZE / 2.0f, IMG_SIZE / 2.0f), 45.0, 1.0);
    cv::cuda::warpAffine(input_gpu, rotated_gpu, affine, cv::Size(IMG_SIZE, IMG_SIZE));

    // 3. 各タイルを非同期に処理
    int tile_idx = 0;
    for (int y = 0; y < IMG_SIZE; y += TILE_SIZE) {
        for (int x = 0; x < IMG_SIZE; x += TILE_SIZE) {
            cv::Rect roi(x, y, TILE_SIZE, TILE_SIZE);
            cv::cuda::GpuMat tile = rotated_gpu(roi);

            // 必要なら float に変換
            cv::cuda::GpuMat tile_float;
            tile.convertTo(tile_float, CV_32F, 1.0, 0.0, streams[tile_idx]);

            // フィルタを作成（外でキャッシュしてもOK）
            auto sobelX = cv::cuda::createSobelFilter(tile_float.type(), -1, 1, 0);
            auto sobelY = cv::cuda::createSobelFilter(tile_float.type(), -1, 0, 1);

            // 非同期適用
            sobelX->apply(tile_float, sobel_xs[tile_idx], streams[tile_idx]);
            sobelY->apply(tile_float, sobel_ys[tile_idx], streams[tile_idx]);

            // 勾配合成
            cv::cuda::magnitude(sobel_xs[tile_idx], sobel_ys[tile_idx], magnitudes[tile_idx], streams[tile_idx]);

            // FFT（結果は複素数になる）
            cv::cuda::dft(magnitudes[tile_idx], fft_results[tile_idx], magnitudes[tile_idx].size(), streams[tile_idx]);

            tile_idx++;
        }
    }

    // 4. 完了待ち
    for (auto& s : streams) s.waitForCompletion();

    // fft_results に結果が16枚格納されています（GPU上）
}


