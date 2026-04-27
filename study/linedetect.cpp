#include <cuda_runtime.h>
#include <cufft.h>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <cstdio>
#include <vector>

// ============================================================
// 切り替えスイッチ
//   USE_CUFFTDX : cuFFTDx版（gatherTiles+FFTを1カーネルに統合）
//   USE_WARP_REDUCTION : findPeakをwarpリダクション版に
// ============================================================
// #define USE_CUFFTDX       // cuFFTDxインストール済みの場合に有効化
#define USE_WARP_REDUCTION   // SM使用率を上げる場合に有効化

#ifdef USE_CUFFTDX
#include <cufftdx.hpp>

// Blackwell sm_120 向け 128点C2C FFT定義
using FFT_1D = decltype(
    cufftdx::Size<128>()
    + cufftdx::Type<cufftdx::fft_type::c2c>()
    + cufftdx::Direction<cufftdx::fft_direction::forward>()
    + cufftdx::Precision<float>()
    + cufftdx::SM<120>()
    + cufftdx::Block()
);

using IFFT_1D = decltype(
    cufftdx::Size<128>()
    + cufftdx::Type<cufftdx::fft_type::c2c>()
    + cufftdx::Direction<cufftdx::fft_direction::inverse>()
    + cufftdx::Precision<float>()
    + cufftdx::SM<120>()
    + cufftdx::Block()
);
#endif

// ============================================================
// 定数
// ============================================================
static constexpr int IMG_W       = 2048;
static constexpr int IMG_H       = 2048;
static constexpr int TILE_W      = 128;   // タイル横サイズ
static constexpr int TILE_H      = 128;   // タイル縦サイズ
static constexpr int STRIDE_X    = 64;    // 横ストライド（TILE_W/2で50%オーバーラップ）
static constexpr int STRIDE_Y    = 64;    // 縦ストライド
static constexpr int NUM_TILES_X = (IMG_W - TILE_W) / STRIDE_X + 1;  // 水平タイル数
static constexpr int NUM_TILES_Y = (IMG_H - TILE_H) / STRIDE_Y + 1;  // 垂直タイル数
static constexpr int NUM_TILES   = NUM_TILES_X * NUM_TILES_Y;
static constexpr int NUM_PAIRS   = (NUM_TILES_X / 2) * NUM_TILES_Y;  // 左右ペア数
static constexpr int RING_SIZE   = 4;

// ============================================================
// 結果構造体
// ============================================================
struct Peak {
    float x;
    float y;
    float val;
};

struct Result {
    int  slot;
    Peak peaks[NUM_PAIRS];
};

// ============================================================
// スレッドセーフキュー
// ============================================================
template<typename T>
class TSQueue {
    std::queue<T>           q;
    std::mutex              m;
    std::condition_variable cv;
public:
    void push(const T& v) {
        std::lock_guard<std::mutex> lk(m);
        q.push(v);
        cv.notify_one();
    }
    T pop() {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&]{ return !q.empty(); });
        T v = q.front(); q.pop();
        return v;
    }
};

// ============================================================
// リングバッファスロット
// ============================================================
struct RingSlot {
    uint16_t*   cpu_ptr        = nullptr;  // Pinned CPUバッファ
    uint16_t*   d_raw          = nullptr;  // GPU入力（uint16）
    float*      d_hann_w       = nullptr;  // ハン窓テーブル横 [TILE_W]
    float*      d_hann_h       = nullptr;  // ハン窓テーブル縦 [TILE_H]
    float2*     d_tiles        = nullptr;  // タイルバッファ [NUM_TILES][TILE_W][TILE_W]
    float2*     d_corr         = nullptr;  // 位相相関結果  [NUM_PAIRS][TILE_W][TILE_W]
    Peak*       d_peaks        = nullptr;  // ピーク座標    [NUM_PAIRS]
    cudaEvent_t doneEvent      = nullptr;
    bool        externalPinned = false;
};

// ============================================================
// CPUでタイル並び替え（USE_CUFFTDXなしの場合のみ使用）
// ============================================================
#ifndef USE_CUFFTDX


// ============================================================
// cuFFT版：Sobel + タイル収集 + ハン窓 + float2変換 を1カーネルで
// ============================================================
__global__ void sobelAndGather(
    float2*         dst,        // [NUM_TILES][TILE_H][TILE_W]
    const uint16_t* src,        // 元画像 [IMG_H][IMG_W]
    const float*    hannW,      // [TILE_W] 横方向ハン窓
    const float*    hannH,      // [TILE_H] 縦方向ハン窓
    int imgW, int imgH, int tileW, int tileH, int numTilesX)
{
    int tileIdx = blockIdx.z;
    int tx = tileIdx % numTilesX;
    int ty = tileIdx / numTilesX;
    int x  = blockIdx.x * blockDim.x + threadIdx.x;
    int y  = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= tileW || y >= tileH) return;

    // 元画像上の絶対座標（ストライドで位置を決める）
    int gx = tx * STRIDE_X + x;
    int gy = ty * STRIDE_Y + y;

    // 境界クランプ
    int x0 = max(gx-1, 0), x1 = min(gx+1, imgW-1);
    int y0 = max(gy-1, 0), y1 = min(gy+1, imgH-1);

    // 3x3近傍を読む
    float p00 = src[y0*imgW+x0], p10 = src[y0*imgW+gx], p20 = src[y0*imgW+x1];
    float p01 = src[gy*imgW+x0],                         p21 = src[gy*imgW+x1];
    float p02 = src[y1*imgW+x0], p12 = src[y1*imgW+gx], p22 = src[y1*imgW+x1];

    // Sobel勾配
    float sx = -p00 + p20 - 2.f*p01 + 2.f*p21 - p02 + p22;
    float sy = -p00 - 2.f*p10 - p20 + p02 + 2.f*p12 + p22;
    float mag = sqrtf(sx*sx + sy*sy);

    // 横・縦独立のハン窓 + float2変換（タイル順に書き出し）
    float w = hannW[x] * hannH[y];
    int dstIdx = tileIdx * tileW * tileH + y * tileW + x;
    dst[dstIdx] = { mag * w, 0.f };
}

#else // USE_CUFFTDX

// ============================================================
// cuFFTDx版：タイル収集 + ハン窓 + FFT を1カーネルで実行
// ============================================================
__global__ void gatherHannAndFFT(
    float2*          dst,        // [NUM_TILES][TILE_H][TILE_W]
    const uint16_t*  src,        // 元画像 [IMG_H][IMG_W]（並び替え不要）
    const float*     hannW,      // [TILE_W] 横方向ハン窓
    const float*     hannH,      // [TILE_H] 縦方向ハン窓
    int imgW, int tileW, int tileH, int numTilesX)
{
    int tileIdx = blockIdx.x;
    int tx = tileIdx % numTilesX;
    int ty = tileIdx / numTilesX;

    __shared__ float2 smem[FFT_1D::shared_memory_size];

    float2* tileOut = dst + tileIdx * tileW * tileH;

    // ① 行方向FFT（TILE_W点）
    for (int row = threadIdx.y; row < tileH; row += blockDim.y) {
        float2 data[FFT_1D::elements_per_thread];
        for (int k = 0; k < FFT_1D::elements_per_thread; k++) {
            int x      = threadIdx.x + k * blockDim.x;
            int srcIdx = (ty * STRIDE_Y + row) * imgW + tx * STRIDE_X + x;
            float w    = hannW[x] * hannH[row];
            data[k]    = { (float)src[srcIdx] * w, 0.f };
        }
        FFT_1D().execute(data, smem);
        for (int k = 0; k < FFT_1D::elements_per_thread; k++) {
            int x = threadIdx.x + k * blockDim.x;
            tileOut[row * tileW + x] = data[k];
        }
    }
    __syncthreads();

    // ② 列方向FFT（TILE_H点）
    for (int col = threadIdx.x; col < tileW; col += blockDim.x) {
        float2 data[IFFT_1D::elements_per_thread];
        for (int k = 0; k < IFFT_1D::elements_per_thread; k++) {
            int y  = threadIdx.y + k * blockDim.y;
            data[k] = tileOut[y * tileW + col];
        }
        IFFT_1D().execute(data, smem);
        for (int k = 0; k < IFFT_1D::elements_per_thread; k++) {
            int y = threadIdx.y + k * blockDim.y;
            tileOut[y * tileW + col] = data[k];
        }
    }
}

// ============================================================
// cuFFTDx版：逆FFT（位相相関結果に適用）
// ============================================================
__global__ void inverseFFT2D(
    float2* data,   // [NUM_PAIRS][TILE_W][TILE_W] in-place
    int tileW)
{
    int pairIdx = blockIdx.x;
    float2* tile = data + pairIdx * tileW * tileW;

    __shared__ float2 smem[IFFT_1D::shared_memory_size];

    // ① 行方向IFFT
    for (int row = threadIdx.y; row < tileW; row += blockDim.y) {
        float2 d[IFFT_1D::elements_per_thread];
        for (int k = 0; k < IFFT_1D::elements_per_thread; k++) {
            int x = threadIdx.x + k * blockDim.x;
            d[k]  = tile[row * tileW + x];
        }
        IFFT_1D().execute(d, smem);
        for (int k = 0; k < IFFT_1D::elements_per_thread; k++) {
            int x = threadIdx.x + k * blockDim.x;
            tile[row * tileW + x] = d[k];
        }
    }
    __syncthreads();

    // ② 列方向IFFT
    for (int col = threadIdx.x; col < tileW; col += blockDim.x) {
        float2 d[IFFT_1D::elements_per_thread];
        for (int k = 0; k < IFFT_1D::elements_per_thread; k++) {
            int y = threadIdx.y + k * blockDim.y;
            d[k]  = tile[y * tileW + col];
        }
        IFFT_1D().execute(d, smem);
        for (int k = 0; k < IFFT_1D::elements_per_thread; k++) {
            int y = threadIdx.y + k * blockDim.y;
            tile[y * tileW + col] = d[k];
        }
    }
}
#endif // USE_CUFFTDX

// ============================================================
// 位相相関カーネル（共通）
// ============================================================
__global__ void phaseCorrelation(
    float2*       out,
    const float2* tiles,
    int tileSize, int numTilesX, int numTilesX2)  // numTilesX2 = numTilesX/2
{
    int pairIdx  = blockIdx.y;
    int col      = pairIdx % numTilesX2;
    int row      = pairIdx / numTilesX2;
    int leftIdx  = row * numTilesX + col;
    int rightIdx = row * numTilesX + col + numTilesX2;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tileSize) return;

    float2 a = tiles[leftIdx  * tileSize + idx];
    float2 b = tiles[rightIdx * tileSize + idx];

    float2 prod = { a.x*b.x + a.y*b.y,
                    a.y*b.x - a.x*b.y };

    float mag = sqrtf(prod.x*prod.x + prod.y*prod.y) + 1e-8f;
    out[pairIdx * tileSize + idx] = { prod.x/mag, prod.y/mag };
}

// ============================================================
// ピーク探索カーネル（共通）
// ============================================================
#ifndef USE_WARP_REDUCTION
// シンプル版
__global__ void findPeak(
    Peak* peaks, const float2* corr, int tileW, int tileH, int tileSize)
{
    int pairIdx = blockIdx.x;
    const float2* tile = corr + pairIdx * tileSize;
    if (threadIdx.x != 0) return;

    float maxVal = -1.f;
    int   maxIdx = 0;
    for (int i = 0; i < tileSize; i++) {
        float v = tile[i].x * tile[i].x + tile[i].y * tile[i].y;
        if (v > maxVal) { maxVal = v; maxIdx = i; }
    }

    int peakX = maxIdx % tileW;
    int peakY = maxIdx / tileW;

    float m00 = 0.f, m10 = 0.f, m01 = 0.f;
    for (int dy = -2; dy <= 2; dy++)
    for (int dx = -2; dx <= 2; dx++) {
        int nx = (peakX + dx + tileW) % tileW;
        int ny = (peakY + dy + tileH) % tileH;
        float2 c = tile[ny * tileW + nx];
        float  v = c.x * c.x + c.y * c.y;
        m00 += v; m10 += v * dx; m01 += v * dy;
    }

    float subX = peakX + m10 / m00;
    float subY = peakY + m01 / m00;

    if (subX >= tileW / 2) subX -= tileW;
    if (subY >= tileH / 2) subY -= tileH;

    peaks[pairIdx].x   = subX;
    peaks[pairIdx].y   = subY;
    peaks[pairIdx].val = m00;
}

#else
// warpリダクション版
__global__ void findPeak(
    Peak* peaks, const float2* corr, int tileW, int tileH, int tileSize)
{
    int pairIdx = blockIdx.x;
    int tid     = threadIdx.x;
    const float2* tile = corr + pairIdx * tileSize;

    float maxVal = -1.f;
    int   maxIdx = 0;
    for (int i = tid; i < tileSize; i += blockDim.x) {
        float v = tile[i].x * tile[i].x + tile[i].y * tile[i].y;
        if (v > maxVal) { maxVal = v; maxIdx = i; }
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float otherVal = __shfl_down_sync(0xffffffff, maxVal, offset);
        int   otherIdx = __shfl_down_sync(0xffffffff, maxIdx, offset);
        if (otherVal > maxVal) { maxVal = otherVal; maxIdx = otherIdx; }
    }

    int warpId   = tid / warpSize;
    int laneId   = tid % warpSize;
    int numWarps = blockDim.x / warpSize;

    __shared__ float sVal[32];
    __shared__ int   sIdx[32];

    if (laneId == 0) { sVal[warpId] = maxVal; sIdx[warpId] = maxIdx; }
    __syncthreads();

    if (warpId == 0) {
        maxVal = laneId < numWarps ? sVal[laneId] : -1.f;
        maxIdx = laneId < numWarps ? sIdx[laneId] : 0;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            float otherVal = __shfl_down_sync(0xffffffff, maxVal, offset);
            int   otherIdx = __shfl_down_sync(0xffffffff, maxIdx, offset);
            if (otherVal > maxVal) { maxVal = otherVal; maxIdx = otherIdx; }
        }
        if (laneId == 0) {
            // 生の[0, N-1]空間でピーク位置を取得
            int peakX = maxIdx % tileW;
            int peakY = maxIdx / tileW;

            float m00 = 0.f, m10 = 0.f, m01 = 0.f;
            for (int dy = -2; dy <= 2; dy++)
            for (int dx = -2; dx <= 2; dx++) {
                int nx = (peakX + dx + tileW) % tileW;
                int ny = (peakY + dy + tileH) % tileH;
                float2 c = tile[ny * tileW + nx];
                float  v = c.x * c.x + c.y * c.y;
                m00 += v; m10 += v * dx; m01 += v * dy;
            }

            float subX = peakX + m10 / m00;
            float subY = peakY + m01 / m00;

            if (subX >= tileW / 2) subX -= tileW;
            if (subY >= tileH / 2) subY -= tileH;

            peaks[pairIdx].x   = subX;
            peaks[pairIdx].y   = subY;
            peaks[pairIdx].val = m00;
        }
    }
}
#endif // USE_WARP_REDUCTION

// ============================================================
// パイプライン本体
// ============================================================
class CudaPipeline {
public:

    bool init() {
        for (int i = 0; i < RING_SIZE; i++) {
            cudaStreamCreate(&streams_[i]);
            if (!initSlot(i))   return false;
            if (!buildGraph(i)) return false;
            freeSlots_.push(i);
        }
        return true;
    }

    // 外部バッファをPinned登録（カメラSDK等が確保済みの場合）
    bool registerExternalRingBuffer(void* ptrs[], int count, size_t sizeEach) {
        if (count != RING_SIZE) {
            fprintf(stderr, "registerExternalRingBuffer: count must be %d\n", RING_SIZE);
            return false;
        }
        for (int i = 0; i < RING_SIZE; i++) {
            auto& s = slots_[i];
            if (s.cpu_ptr && !s.externalPinned) {
                cudaFreeHost(s.cpu_ptr);
                s.cpu_ptr = nullptr;
            }
            cudaError_t err = cudaHostRegister(ptrs[i], sizeEach, cudaHostRegisterDefault);
            if (err != cudaSuccess) {
                fprintf(stderr, "cudaHostRegister slot[%d] failed: %s\n",
                        i, cudaGetErrorString(err));
                return false;
            }
            s.cpu_ptr        = static_cast<uint16_t*>(ptrs[i]);
            s.externalPinned = true;
        }
        return true;
    }

    // GPUクロックを最大に固定（管理者権限が必要）
    void lockGpuClock() {
        nvmlInit();
        nvmlDevice_t nvmlDev;
        nvmlDeviceGetHandleByIndex(0, &nvmlDev);
        unsigned int minClock, maxClock;
        nvmlDeviceGetMinMaxClockOfPState(nvmlDev, NVML_CLOCK_GRAPHICS,
            NVML_PSTATE_0, &minClock, &maxClock);
        nvmlDeviceSetGpuLockedClocks(nvmlDev, maxClock, maxClock);
        nvmlShutdown();
    }

    void unlockGpuClock() {
        nvmlInit();
        nvmlDevice_t nvmlDev;
        nvmlDeviceGetHandleByIndex(0, &nvmlDev);
        nvmlDeviceResetGpuLockedClocks(nvmlDev);
        nvmlShutdown();
    }

    // 各スレッドから直接呼ぶ（スレッドセーフ）
    Result process(const void* cameraData) {
        int slot = freeSlots_.pop();
        auto& s  = slots_[slot];

#ifndef USE_CUFFTDX
        // cuFFT版：元画像をそのままコピー
        memcpy(s.cpu_ptr, cameraData, IMG_W * IMG_H * sizeof(uint16_t));
#else
        // cuFFTDx版：並び替え不要、そのままコピー
        memcpy(s.cpu_ptr, cameraData, IMG_W * IMG_H * sizeof(uint16_t));
#endif

        cudaGraphLaunch(graphExecs_[slot], streams_[slot]);
        cudaEventRecord(s.doneEvent, streams_[slot]);
        cudaEventSynchronize(s.doneEvent);

        Result result;
        result.slot = slot;
        cudaMemcpy(result.peaks, s.d_peaks,
                   NUM_PAIRS * sizeof(Peak), cudaMemcpyDeviceToHost);  // 左右ペア分

        freeSlots_.push(slot);
        return result;
    }

    void cleanup() {
        for (int i = 0; i < RING_SIZE; i++) {
            if (graphExecs_[i]) cudaGraphExecDestroy(graphExecs_[i]);
            if (graphs_[i])     cudaGraphDestroy(graphs_[i]);
            if (streams_[i])    cudaStreamDestroy(streams_[i]);

            auto& s = slots_[i];
            if (s.cpu_ptr) {
                if (s.externalPinned) cudaHostUnregister(s.cpu_ptr);
                else                  cudaFreeHost(s.cpu_ptr);
            }
            if (s.d_raw)        cudaFree(s.d_raw);
            if (s.d_hann_w) cudaFree(s.d_hann_w);
            if (s.d_hann_h) cudaFree(s.d_hann_h);
            if (s.d_tiles) cudaFree(s.d_tiles);
            if (s.d_corr)  cudaFree(s.d_corr);
            if (s.d_peaks) cudaFree(s.d_peaks);
            if (s.doneEvent) cudaEventDestroy(s.doneEvent);
        }
#ifndef USE_CUFFTDX
        for (int i = 0; i < RING_SIZE; i++) {
            if (plans_[i])  cufftDestroy(plans_[i]);
            if (iplans_[i]) cufftDestroy(iplans_[i]);
        }
#endif
    }

private:
    RingSlot        slots_[RING_SIZE];
    cudaStream_t    streams_[RING_SIZE]    = {};
    cudaGraph_t     graphs_[RING_SIZE]     = {};
    cudaGraphExec_t graphExecs_[RING_SIZE] = {};
    TSQueue<int>    freeSlots_;

#ifndef USE_CUFFTDX
    cufftHandle     plans_[RING_SIZE]      = {};  // Forward FFT
    cufftHandle     iplans_[RING_SIZE]     = {};  // Inverse FFT
#endif

    bool initSlot(int i) {
        auto& s = slots_[i];

        // CPUバッファ（Pinned）
        size_t cpuSize = IMG_W * IMG_H * sizeof(uint16_t);
        cudaMallocHost(&s.cpu_ptr, cpuSize);

#ifndef USE_CUFFTDX
        // cuFFT版：元画像バッファ
        cudaMalloc(&s.d_raw, IMG_W * IMG_H * sizeof(uint16_t));
#else
        // cuFFTDx版：元画像そのままのバッファ
        cudaMalloc(&s.d_raw, IMG_W * IMG_H * sizeof(uint16_t));
#endif

        // ハン窓テーブル（横・縦、初期化時に1回だけ計算）
        cudaMalloc(&s.d_hann_w, TILE_W * sizeof(float));
        cudaMalloc(&s.d_hann_h, TILE_H * sizeof(float));
        std::vector<float> hannW(TILE_W), hannH(TILE_H);
        for (int j = 0; j < TILE_W; j++)
            hannW[j] = 0.5f - 0.5f * cosf(2.f * M_PI * j / (TILE_W - 1));
        for (int j = 0; j < TILE_H; j++)
            hannH[j] = 0.5f - 0.5f * cosf(2.f * M_PI * j / (TILE_H - 1));
        cudaMemcpy(s.d_hann_w, hannW.data(), TILE_W * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(s.d_hann_h, hannH.data(), TILE_H * sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc(&s.d_tiles, NUM_TILES * TILE_W * TILE_H * sizeof(float2));
        cudaMalloc(&s.d_corr,  NUM_PAIRS * TILE_W * TILE_H * sizeof(float2));
        cudaMalloc(&s.d_peaks, NUM_PAIRS * sizeof(Peak));
        cudaEventCreate(&s.doneEvent);
        return true;
    }

    bool buildGraph(int i) {
        auto& s = slots_[i];

#ifndef USE_CUFFTDX
        // cuFFT版：planを作成
        int n[2] = { TILE_H, TILE_W };  // 行優先（H, W）
        if (cufftPlanMany(&plans_[i], 2, n,
                          nullptr, 1, TILE_W * TILE_H,
                          nullptr, 1, TILE_W * TILE_H,
                          CUFFT_C2C, NUM_TILES) != CUFFT_SUCCESS) {
            fprintf(stderr, "cufftPlanMany fwd slot[%d] failed\n", i);
            return false;
        }
        cufftSetStream(plans_[i], streams_[i]);

        if (cufftPlanMany(&iplans_[i], 2, n,
                          nullptr, 1, TILE_W * TILE_H,
                          nullptr, 1, TILE_W * TILE_H,
                          CUFFT_C2C, NUM_PAIRS) != CUFFT_SUCCESS) {
            fprintf(stderr, "cufftPlanMany inv slot[%d] failed\n", i);
            return false;
        }
        cufftSetStream(iplans_[i], streams_[i]);
#endif

        cudaStreamBeginCapture(streams_[i], cudaStreamCaptureModeGlobal);

#ifndef USE_CUFFTDX
        // ============================================================
        // cuFFT版フロー
        // ① H2D転送（タイル並び替え済み、16MB）

        // ③ Forward FFT（cuFFT 256バッチ）
        // ④ 位相相関
        // ⑤ Inverse FFT（cuFFT 128バッチ）
        // ⑥ findPeak
        // ============================================================

        // ① H2D転送（元画像uint16、32MB）
        cudaMemcpyAsync(s.d_raw, s.cpu_ptr,
                        IMG_W * IMG_H * sizeof(uint16_t),
                        cudaMemcpyHostToDevice, streams_[i]);

        // ② Sobel + タイル収集 + ハン窓（1カーネルで完結）
        dim3 sgBlock(16, 16);
        dim3 sgGrid((TILE_W + 15) / 16, (TILE_H + 15) / 16, NUM_TILES);
        sobelAndGather<<<sgGrid, sgBlock, 0, streams_[i]>>>(
            s.d_tiles, s.d_raw, s.d_hann_w, s.d_hann_h,
            IMG_W, IMG_H, TILE_W, TILE_H, NUM_TILES_X);

        // ③ Forward FFT
        cufftExecC2C(plans_[i], s.d_tiles, s.d_tiles, CUFFT_FORWARD);

        // ④ 位相相関
        dim3 corrBlock(256);
        dim3 corrGrid((TILE_W * TILE_H + 255) / 256, NUM_PAIRS);
        phaseCorrelation<<<corrGrid, corrBlock, 0, streams_[i]>>>(
            s.d_corr, s.d_tiles, TILE_W * TILE_H, NUM_TILES_X, NUM_TILES_X / 2);

        // ⑤ Inverse FFT
        cufftExecC2C(iplans_[i], s.d_corr, s.d_corr, CUFFT_INVERSE);

#else
        // ============================================================
        // cuFFTDx版フロー
        // ① H2D転送（元画像そのまま、16MB）
        // ② gatherHannAndFFT（gather + ハン窓 + FFT を1カーネルで）
        // ③ 位相相関
        // ④ inverseFFT2D（cuFFTDxで逆FFT）
        // ⑤ findPeak
        // ============================================================

        // ① H2D転送
        cudaMemcpyAsync(s.d_raw, s.cpu_ptr,
                        IMG_W * IMG_H * sizeof(uint16_t),
                        cudaMemcpyHostToDevice, streams_[i]);

        // ② gatherHannAndFFT（1カーネルで完結）
        gatherHannAndFFT<<<NUM_TILES, FFT_1D::block_dim, 0, streams_[i]>>>(
            s.d_tiles, s.d_raw, s.d_hann_w, s.d_hann_h,
            IMG_W, TILE_W, TILE_H, NUM_TILES_X);

        // ③ 位相相関
        dim3 corrBlock(256);
        dim3 corrGrid((TILE_W * TILE_H + 255) / 256, NUM_PAIRS);
        phaseCorrelation<<<corrGrid, corrBlock, 0, streams_[i]>>>(
            s.d_corr, s.d_tiles, TILE_W * TILE_H, NUM_TILES_X, NUM_TILES_X / 2);

        // ④ Inverse FFT
        inverseFFT2D<<<NUM_PAIRS, IFFT_1D::block_dim, 0, streams_[i]>>>(
            s.d_corr, TILE_W);

#endif // USE_CUFFTDX

        // ⑥ findPeak（共通）
#ifndef USE_WARP_REDUCTION
        findPeak<<<NUM_PAIRS, 1, 0, streams_[i]>>>(
            s.d_peaks, s.d_corr, TILE_W, TILE_H, TILE_W * TILE_H);
#else
        findPeak<<<NUM_PAIRS, 256, 0, streams_[i]>>>(
            s.d_peaks, s.d_corr, TILE_W, TILE_H, TILE_W * TILE_H);
#endif

        cudaStreamEndCapture(streams_[i], &graphs_[i]);
        cudaGraphInstantiate(&graphExecs_[i], graphs_[i], nullptr, nullptr, 0);

        return true;
    }
};

// ============================================================
// 使用例
// ============================================================
//
// // 切り替えはファイル先頭の#defineだけ
// // #define USE_CUFFTDX       → cuFFTDx版（並び替え不要）
// // #define USE_WARP_REDUCTION → findPeakをwarpリダクション版に
//
// CudaPipeline pipeline;
// pipeline.init();
// pipeline.lockGpuClock();
//
// // 各スレッドからそのまま呼ぶ（スロット管理は内部で自動）
// void imageProcessThread() {
//     while (running) {
//         void* cameraData = waitCameraFrame();
//         Result result = pipeline.process(cameraData);
//     }
// }
//
// pipeline.unlockGpuClock();
// pipeline.cleanup();