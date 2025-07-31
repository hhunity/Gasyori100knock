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


