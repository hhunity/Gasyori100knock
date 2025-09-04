
using System;
using System.Drawing;
using System.Windows.Forms;

namespace WinFormsApp
{
    public partial class Form1 : Form
    {
        private Point _lastClickedPointClient = Point.Empty;   // クリック保存用（クライアント座標）

        private readonly ContextMenuStrip _cmenu = new ContextMenuStrip();
        private readonly ToolStripMenuItem _miUsePoint = new ToolStripMenuItem("この座標を入力");

        private readonly PictureBox pictureBox1 = new PictureBox();
        private readonly TextBox textBoxPoint = new TextBox();

        public Form1()
        {
            InitializeComponent();

            // --- PictureBox の初期設定 ---
            pictureBox1.Dock = DockStyle.Fill;
            pictureBox1.BackColor = Color.LightGray;
            pictureBox1.ContextMenuStrip = _cmenu;
            this.Controls.Add(pictureBox1);

            // --- TextBox の初期設定 ---
            textBoxPoint.Dock = DockStyle.Bottom;
            this.Controls.Add(textBoxPoint);

            // --- コンテキストメニュー構築 ---
            _cmenu.Items.Add(_miUsePoint);

            // 右クリック位置を記録
            pictureBox1.MouseDown += (s, e) =>
            {
                if (e.Button == MouseButtons.Right)
                {
                    // pictureBox1 内の相対座標を保存
                    _lastClickedPointClient = e.Location;
                }
            };

            // メニューのコマンドで TextBox に書き込む
            _miUsePoint.Click += (s, e) =>
            {
                textBoxPoint.Text = $"{_lastClickedPointClient.X}, {_lastClickedPointClient.Y}";
            };

            // 念のため、メニューが開く直前に最新座標を再取得
            _cmenu.Opening += (s, e) =>
            {
                if (_cmenu.SourceControl is Control src)
                {
                    var client = src.PointToClient(Cursor.Position);
                    _lastClickedPointClient = client;
                }
            };
        }
    }
}
public class SafeStore
{
    private readonly object _gate = new object(); // 絶対に public にしない & this には lock しない

    private int _value;

    public void Write(int v)
    {
        lock (_gate)
        {
            _value = v;
        }
    }

    public int ReadAndClear()
    {
        lock (_gate)
        {
            int tmp = _value;
            _value = 0;
            return tmp;
        }
    }
}

using var mtx = new Mutex(false, @"Global\MyApp_UniqueResource");
mtx.WaitOne();
try
{
    // 共有資源を使用
}
finally { mtx.ReleaseMutex(); }








// GpuCtx にメンバ追加
cufftHandle planC2C4 = 0;
int tileW = 0, tileH = 0; // プランのサイズを覚えておく


// gp_create_ctx の最後あたり（回転行列などの後）
{
    ctx->tileH = height;
    ctx->tileW = width / 4;                 // W%4==0 前提

    int n[2]      = { ctx->tileW, ctx->tileH };  // {X=幅, Y=高さ}
    int inembed[2]= { ctx->tileW, ctx->tileH };
    int onembed[2]= { ctx->tileW, ctx->tileH };
    int istride   = 1, ostride = 1;
    int idist     = ctx->tileW * ctx->tileH;     // 1スライスの要素数
    int odist     = idist;
    int batch     = 4;

    cufftPlanMany(&ctx->planC2C4, 2, n,
                  inembed, istride, idist,
                  onembed, ostride, odist,
                  CUFFT_C2C, batch);
    // FFTは回転完了後の sFFT ストリームで回す想定（実行時に SetStream）
}

#include "hann_window.hpp"
#include <opencv2/core.hpp>  // createHanningWindow

// GpuCtx 側にキャッシュを持たせる想定
struct GpuCtx {
    std::mutex winMu;
    // 幅だけが変わる運用なら key=W で十分。高さも変わるなら pair<int,int> をキーにする。
    std::unordered_map<int, cv::cuda::GpuMat> winCacheGpu;
};

cv::cuda::GpuMat& getHann2D(GpuCtx* ctx, int H, int W)
{
    std::lock_guard<std::mutex> lk(ctx->winMu);

    if (auto it = ctx->winCacheGpu.find(W); it != ctx->winCacheGpu.end())
        return it->second;

    // CPUで2D Hann生成（OpenCVが外積で作ってくれる）
    cv::Mat hann2d;
    cv::createHanningWindow(hann2d, cv::Size(W, H), CV_32F); // H×W, CV_32FC1

    // GPUへアップロードしてキャッシュ
    cv::cuda::GpuMat winGpu;
    winGpu.upload(hann2d);

    auto [it2, ok] = ctx->winCacheGpu.emplace(W, std::move(winGpu));
    return it2->second;
}

// 2D Hann (H×w) を GPU にキャッシュ（前に出したやつでOK）
cv::cuda::GpuMat& getHann2D(GpuCtx* ctx, int H, int w);

// 実数タイルを float2(Imag=0) へ詰める＋窓を掛けるカーネル
__global__ void pack_with_window(const float* __restrict__ src, size_t srcPitchFloats,
                                 const float* __restrict__ win, // H×w
                                 float2* __restrict__ dst, int w, int H, size_t dstSliceElems)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= w || y >= H) return;
    float v = src[y*srcPitchFloats + x] * win[y*w + x];
    dst[y*w + x] = make_float2(v, 0.0f);
    // dst は各バッチの先頭を渡す想定（呼び出し側で +t*dstSliceElems する）
}


// 2D Hann (H×w) を GPU にキャッシュ（前に出したやつでOK）
cv::cuda::GpuMat& getHann2D(GpuCtx* ctx, int H, int w);

// 実数タイルを float2(Imag=0) へ詰める＋窓を掛けるカーネル
__global__ void pack_with_window(const float* __restrict__ src, size_t srcPitchFloats,
                                 const float* __restrict__ win, // H×w
                                 float2* __restrict__ dst, int w, int H, size_t dstSliceElems)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= w || y >= H) return;
    float v = src[y*srcPitchFloats + x] * win[y*w + x];
    dst[y*w + x] = make_float2(v, 0.0f);
    // dst は各バッチの先頭を渡す想定（呼び出し側で +t*dstSliceElems する）
}

// 2D Hann (H×w) を GPU にキャッシュ（前に出したやつでOK）
cv::cuda::GpuMat& getHann2D(GpuCtx* ctx, int H, int w);

// 実数タイルを float2(Imag=0) へ詰める＋窓を掛けるカーネル
__global__ void pack_with_window(const float* __restrict__ src, size_t srcPitchFloats,
                                 const float* __restrict__ win, // H×w
                                 float2* __restrict__ dst, int w, int H, size_t dstSliceElems)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= w || y >= H) return;
    float v = src[y*srcPitchFloats + x] * win[y*w + x];
    dst[y*w + x] = make_float2(v, 0.0f);
    // dst は各バッチの先頭を渡す想定（呼び出し側で +t*dstSliceElems する）
}

const int H = ctx->H, W = ctx->W;
const int batch = 4;
const int w = ctx->tileW;     // W/4
const size_t sliceElems = (size_t)H * w;

// 連結出力（CV_32FC2, H×W）をGPUに確保
s.d_fft_cat.create(H, W, CV_32FC2);

// batched 入力バッファ（float2, [4][H][w]）を一時確保（スロット再利用推奨）
float2* d_batch = nullptr;
cudaMallocAsync(&d_batch, batch * sliceElems * sizeof(float2),
                cv::cuda::StreamAccessor::getStream(sK));  // sK/sFFTどちらでも

// FFT ストリームを決める（回転に依存）
cv::cuda::Stream sFFT = sK;
sFFT.waitEvent(s.evK);
cufftSetStream(ctx->planC2C4, cv::cuda::StreamAccessor::getStream(sFFT));

// 窓
cv::cuda::GpuMat& win = getHann2D(ctx, H, w);

// 4タイルを pack_with_window で一括パック
dim3 blk(32, 8);
dim3 grd((w + blk.x - 1)/blk.x, (H + blk.y - 1)/blk.y);

for (int t = 0; t < batch; ++t) {
    cv::Rect roi(t * w, 0, w, H);
    cv::cuda::GpuMat tile = s.d_out(roi);     // CV_32FC1
    const float* src = tile.ptr<float>();
    size_t srcPitchF = tile.step / sizeof(float);

    float2* dst = d_batch + (size_t)t * sliceElems;

    pack_with_window<<<grd, blk, 0, cv::cuda::StreamAccessor::getStream(sFFT)>>>(
        src, srcPitchF, win.ptr<float>(), dst, w, H, sliceElems);
}

// batched FFT（in-place）
cufftExecC2C(ctx->planC2C4,
             reinterpret_cast<cufftComplex*>(d_batch),
             reinterpret_cast<cufftComplex*>(d_batch),
             CUFFT_FORWARD);

// batched の各スライスを横に連結して d_fft_cat に配置（GPU内2Dコピー）
for (int t = 0; t < batch; ++t) {
    cv::Rect roi(t * w, 0, w, H);
    // dst: GpuMat ROI (CV_32FC2)
    auto dst = s.d_fft_cat(roi);
    // src: d_batch + t*sliceElems（連続, pitch = w*sizeof(float2)）
    cudaMemcpy2DAsync(dst.ptr(), dst.step,
                      d_batch + (size_t)t * sliceElems, w * sizeof(float2),
                      w * sizeof(float2), H,
                      cudaMemcpyDeviceToDevice,
                      cv::cuda::StreamAccessor::getStream(sFFT));
}

// （ここで d_batch を解放 or 再利用用に Slot に保持）
cudaFreeAsync(d_batch, cv::cuda::StreamAccessor::getStream(sFFT));

// まとめて1回 D2H（CV_32FC2, H×W）
sD2H.waitEvent(s.evK);          // sFFT と同一なら暗黙順序でもOK
s.d_fft_cat.download(s.fft_cat_host, sD2H);  // cv::Mat(CV_32FC2)
cudaEventRecord(s.evD2H, cv::cuda::StreamAccessor::getStream(sD2H));

// D2H 完了 → cudaLaunchHostFunc で軽い通知 → 自前プール or 直接コールバック



// 依存: H2D→回転
sK.waitEvent(s.evH2D);
cv::cuda::warpAffine(s.d_in, s.d_out, ctx->rotM, s.d_out.size(),
                     cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0), sK);
cudaEventRecord(s.evK, cv::cuda::StreamAccessor::getStream(sK));

// ===== ここから窓＋前方DFT（GPU） =====
const int W = ctx->W, H = ctx->H;
const int tiles = 4;
const int baseW = W / tiles;
const int rem   = W % tiles;

// DFT用の出力（複素2ch）をGPUに用意（4帯を横に並べる）
s.d_fft_cat.create(H, W, CV_32FC2);   // <- 新規: Slot に GpuMat 追加しておく

cv::cuda::Stream sFFT = sK;           // 同じでもOK（別にしても可）
sFFT.waitEvent(s.evK);

int x = 0;
for (int t = 0; t < tiles; ++t) {
    int w = baseW + ((t == tiles-1) ? rem : 0);
    cv::Rect roi(x, 0, w, H); x += w;

    // 1) ROI（回転後, CV_32FC1）
    cv::cuda::GpuMat tile = s.d_out(roi);

    // 2) 窓（Hann）をGPUで掛ける
    auto& win = getHann2D(ctx, H, w);
    cv::cuda::multiply(tile, win, tile, 1.0, -1, sFFT); // in-place OK

    // 3) 前方2D DFT（パディングなし, 複素出力 CV_32FC2）
    cv::cuda::GpuMat complex; // H×w×2ch
    cv::cuda::dft(tile, complex, cv::Size(), cv::DFT_COMPLEX_OUTPUT, sFFT);

    // 4) 横に連結（GPU内コピー）
    complex.copyTo(s.d_fft_cat(roi), sFFT);
}

// 5) まとめて1回だけ D2H（複素2chのまま）
sD2H.waitEvent(s.evK); //（sFFT と同一なら暗黙順序でOK）
s.d_fft_cat.download(s.fft_cat_host, sD2H); // cv::Mat(CV_32FC2) を Slot に用意しておく
cudaEventRecord(s.evD2H, cv::cuda::StreamAccessor::getStream(sD2H));

// ===== D2H 完了でホスト関数 → コールバック =====
auto rawD2H = cv::cuda::StreamAccessor::getStream(sD2H);
struct Payload { GpuCtx* ctx; int slot; int fid; void* user; };
auto* p = new Payload{ctx, j.slot, j.frame_id, j.user};

cudaLaunchHostFunc(rawD2H, [](void* ud){
    std::unique_ptr<Payload> P((Payload*)ud);
    auto* ctx = P->ctx; int slot = P->slot; int fid = P->fid; void* user = P->user;
    auto& s = ctx->slots[slot];

    // s.fft_cat_host: CV_32FC2, 幅W×高さH, 各画素が (Re,Im)
    if (ctx->cb) {
        ctx->cb(fid,
                reinterpret_cast<const float*>(s.fft_cat_host.ptr()),
                ctx->W, ctx->H,
                static_cast<int>(s.fft_cat_host.step),   // バイト単位の行ストライド
                user ? user : ctx->user);
    }
    // スロット解放
    { std::lock_guard<std::mutex> lk(ctx->mu);
      s.id = -1; ctx->freeSlots.push(slot); }
    ctx->cv.notify_all();
}, p);



using System;
using System.Diagnostics.Tracing;
using System.Threading;

[EventSource(Name = "MyCompany-MyApp")]
class MyEventSource : EventSource
{
    public static readonly MyEventSource Log = new MyEventSource();

    [Event(1, Message = "FFT開始", Level = EventLevel.Informational)]
    public void FftStart() => WriteEvent(1);

    [Event(2, Message = "FFT終了", Level = EventLevel.Informational)]
    public void FftEnd() => WriteEvent(2);
}

class Program
{
    static void Main()
    {
        MyEventSource.Log.FftStart();
        Thread.Sleep(50); // ダミー処理
        MyEventSource.Log.FftEnd();
    }
}

#include <windows.h>
#include <TraceLoggingProvider.h>

// プロバイダー定義
TRACELOGGING_DEFINE_PROVIDER(
    g_hMyProvider,
    "MyCompany-MyApp",
    // GUID は `uuidgen` で生成
    (0x12345678,0x1234,0x1234,0x12,0x34,0x56,0x78,0x9a,0xbc,0xde,0xf0)
);

int main()
{
    TraceLoggingRegister(g_hMyProvider);

    TraceLoggingWrite(g_hMyProvider, "FFT_Start");
    Sleep(50);
    TraceLoggingWrite(g_hMyProvider, "FFT_End");

    TraceLoggingUnregister(g_hMyProvider);
    return 0;
}




#pragma once
#include <nvToolsExt.h>
#include <string>

class NvtxRange
{
public:
    // スコープ自動管理用 (RAII)
    NvtxRange(const char* name, int category = 0, unsigned int argb = 0xFF80C0FF)
    {
        nvtxEventAttributes_t attr = {};
        attr.version = NVTX_VERSION;
        attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        attr.category = category;
        attr.colorType = NVTX_COLOR_ARGB;
        attr.color = argb;
        attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
        attr.message.ascii = name;
        id_ = nvtxRangeStartEx(&attr);
    }

    ~NvtxRange()
    {
        if (id_ != 0)
            nvtxRangeEnd(id_);
    }

    // 非同期用途：明示的に閉じる
    void End()
    {
        if (id_ != 0)
        {
            nvtxRangeEnd(id_);
            id_ = 0;
        }
    }

    // コピー禁止、ムーブ許可
    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
    NvtxRange(NvtxRange&& other) noexcept { id_ = other.id_; other.id_ = 0; }
    NvtxRange& operator=(NvtxRange&& other) noexcept
    {
        if (this != &other)
        {
            End();
            id_ = other.id_;
            other.id_ = 0;
        }
        return *this;
    }

private:
    nvtxRangeId_t id_{0};
};

using System;
using System.Runtime.InteropServices;

internal static class NvtxEx
{
    [StructLayout(LayoutKind.Sequential)]
    private struct nvtxEventAttributes_t
    {
        public ushort version;
        public ushort size;
        public int category;
        public int colorType;
        public uint color;
        public int messageType;
        public IntPtr message;
    }

    [DllImport("nvToolsExt64_1.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern ulong nvtxRangeStartEx(ref nvtxEventAttributes_t attr);

    [DllImport("nvToolsExt64_1.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern void nvtxRangeEnd(ulong id);

    public static ulong Begin(string name, int cat = 0, uint argb = 0xFF80C0FF)
    {
        var bytes = System.Text.Encoding.ASCII.GetBytes(name + "\0");
        var handle = GCHandle.Alloc(bytes, GCHandleType.Pinned);
        try
        {
            var attr = new nvtxEventAttributes_t
            {
                version = 1,
                size = (ushort)Marshal.SizeOf<nvtxEventAttributes_t>(),
                category = cat,
                colorType = 1,
                color = argb,
                messageType = 1,
                message = handle.AddrOfPinnedObject()
            };
            return nvtxRangeStartEx(ref attr);
        }
        finally
        {
            handle.Free();
        }
    }

    public static void End(ulong id)
    {
        nvtxRangeEnd(id);
    }
}


[StructLayout(LayoutKind.Sequential)]
struct nvtxEventAttributes_t {
    public ushort version; public ushort size;
    public int category; public int colorType; public uint color;
    public int messageType; public IntPtr message; // ANSI
}
internal static class NvtxEx {
    [DllImport("nvToolsExt64_1.dll", CallingConvention = CallingConvention.Cdecl)]
    static extern ulong nvtxRangeStartEx(ref nvtxEventAttributes_t attr);
    [DllImport("nvToolsExt64_1.dll", CallingConvention = CallingConvention.Cdecl)]
    static extern void nvtxRangeEnd(ulong id);

    public static IDisposable Push(string name, int cat = 0, uint rgb = 0x00A0FFFF) {
        var bytes = System.Text.Encoding.ASCII.GetBytes(name + "\0");
        var ptr = Marshal.UnsafeAddrOfPinnedArrayElement(bytes, 0);
        var a = new nvtxEventAttributes_t {
            version = 1, size = (ushort)Marshal.SizeOf<nvtxEventAttributes_t>(),
            category = cat, colorType = 1, color = rgb, messageType = 1, message = ptr
        };
        ulong id = nvtxRangeStartEx(ref a);
        return new Pop{id=id};
    }
    private sealed class Pop : IDisposable { public ulong id; public void Dispose()=>nvtxRangeEnd(id); }
}

using System;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

internal static class Nvtx
{
    // Windows x64 の nvToolsExt
    [DllImport("nvToolsExt64_1.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern int nvtxRangePushA([MarshalAs(UnmanagedType.LPStr)] string message);

    [DllImport("nvToolsExt64_1.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern int nvtxRangePop();

    [DllImport("nvToolsExt64_1.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void nvtxNameOsThread(uint threadId, [MarshalAs(UnmanagedType.LPStr)] string name);

    [DllImport("kernel32.dll")]
    private static extern uint GetCurrentThreadId();

    public static IDisposable Push(string name)
    {
        // スレッドに名前が未設定なら付ける（任意）
        nvtxNameOsThread(GetCurrentThreadId(), $"T{GetCurrentThreadId()}");
        nvtxRangePushA(name);
        return new PopOnDispose();
    }

    private sealed class PopOnDispose : IDisposable
    {
        public void Dispose() => nvtxRangePop();
    }
}

// 使い方例：任意区間をNVTXで囲む
public static class Example
{
    public static void Run()
    {
        Parallel.For(0, 8, i =>
        {
            using (Nvtx.Push($"Item {i}"))
            {
                using (Nvtx.Push("Rotate")) Rotate();
                using (Nvtx.Push("FFT"))    Fft();
                using (Nvtx.Push("Post"))   Post();
            }
        });
    }

    static void Rotate() { /* 対象処理 */ Thread.SpinWait(200000); }
    static void Fft()    { /* 対象処理 */ Thread.SpinWait(400000); }
    static void Post()   { /* 対象処理 */ Thread.SpinWait(150000); }
}

// ComputeBackend.h
#pragma once
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <atomic>
#include <mutex>
#include <string>

enum class ComputeBackend { CPU, GPU };

bool InitComputeBackend();         // 起動時に1回呼ぶ
ComputeBackend GetBackend();       // どっちで動くか取得
void TripToCpu(const char* reason); // 途中でGPUが落ちたらCPUへ切替（回路遮断）
bool IsCudaBuild();                // 参考：ビルドがCUDA対応か

// ComputeBackend.cpp
#include "ComputeBackend.h"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>
#include <cstdlib>
#include <iostream>

namespace {
std::once_flag g_once;
std::atomic<ComputeBackend> g_backend{ComputeBackend::CPU};

bool detectCudaOnce() {
    try {
        // 環境変数で強制CPU（デバッグ運用用）
        if (const char* v = std::getenv("APP_FORCE_CPU")) {
            if (std::string(v) == "1") return false;
        }

        // 1) デバイス数
        int n = cv::cuda::getCudaEnabledDeviceCount();
        if (n <= 0) return false;

        // 2) 互換性（Compute Capabilityなど）
        cv::cuda::DeviceInfo info(0);
        if (!info.isCompatible()) return false;

        // 3) 最小確保テスト（ドライバ異常などの早期検出）
        cv::cuda::GpuMat test(8, 8, CV_8UC1);
        (void)test;

        return true;
    } catch (...) {
        return false;
    }
}
} // namespace

bool InitComputeBackend() {
    std::call_once(g_once, [] {
        g_backend.store(detectCudaOnce() ? ComputeBackend::GPU : ComputeBackend::CPU,
                        std::memory_order_relaxed);
        std::cout << "[Init] ComputeBackend = "
                  << (g_backend.load()==ComputeBackend::GPU ? "GPU" : "CPU") << std::endl;
        if (g_backend.load()==ComputeBackend::GPU) {
            try { cv::cuda::printCudaDeviceInfo(0); } catch (...) {}
        }
    });
    return g_backend.load()==ComputeBackend::GPU;
}

ComputeBackend GetBackend() {
    return g_backend.load(std::memory_order_relaxed);
}

void TripToCpu(const char* reason) {
    auto prev = g_backend.exchange(ComputeBackend::CPU);
    if (prev != ComputeBackend::CPU) {
        std::cerr << "[WARN] Switched to CPU due to GPU failure: "
                  << (reason ? reason : "(unknown)") << std::endl;
    }
}

bool IsCudaBuild() {
    try {
        const auto bi = cv::getBuildInformation();
        return bi.find("CUDA") != std::string::npos; // 参考表示用（判定は detectCudaOnce が本体）
    } catch (...) {
        return false;
    }
}



// 依存: 回転完了
cv::cuda::Stream sFFT = sK;
sFFT.waitEvent(s.evK);

const int W = ctx->W, H = ctx->H;
const int tiles = 4;
const int baseW = W / tiles;
const int rem   = W % tiles;

s.d_mag_cat.create(H, W, CV_32FC1);

int x = 0;
for (int t = 0; t < tiles; ++t) {
    int w = baseW + ((t == tiles-1) ? rem : 0);
    cv::Rect roi(x, 0, w, H);
    x += w;

    cv::cuda::GpuMat tile = s.d_out(roi); // 回転後のタイル CV_32FC1

    // ★ ここで窓を掛ける（GPU）
    cv::cuda::GpuMat& winGpu = getHann2D(ctx, H, w);
    cv::cuda::multiply(tile, winGpu, tile, 1.0, -1, sFFT); // tile ← tile * window

    // 2D FFT（複素出力, パディングなし）
    cv::cuda::GpuMat complex; // CV_32FC2
    cv::cuda::dft(tile, complex, cv::Size(), cv::DFT_COMPLEX_OUTPUT, sFFT);

    // magnitude
    std::vector<cv::cuda::GpuMat> planes;
    cv::cuda::split(complex, planes, sFFT);
    cv::cuda::GpuMat mag;
    cv::cuda::magnitude(planes[0], planes[1], mag, sFFT);

    // 横に連結（GPU内）
    mag.copyTo(s.d_mag_cat(roi), sFFT);
}

// まとめて1回 D2H → Host コールバック（前回と同じ）



// gpu_async_fftpool.cpp  — ① 非同期Submit + DLL内 FFT専用スレッドプール + 最終結果コールバック
// ビルド例: cl /O2 /MD /EHsc /LD gpu_async_fftpool.cpp /I<opencv\include> <opencv libs...>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#ifdef _WIN32
  #define DLL_EXPORT __declspec(dllexport)
  #define DLL_CALL   __cdecl
#else
  #define DLL_EXPORT
  #define DLL_CALL
#endif

extern "C" {
// 最終結果（回転→FFTのmagnitude）だけを返す
typedef void (DLL_CALL *ResultCallback)(
    int frameId,
    const float* data,   // 先頭ポインタ（row0）
    int width, int height,
    int strideBytes,     // 1行のバイト数（= width*sizeof(float) が基本）
    void* user           // gp_submit_* の引数 user をそのまま返す
);

// 不透明ハンドル
struct GpuCtx;

// コンテキスト作成/破棄
DLL_EXPORT GpuCtx* DLL_CALL gp_create_ctx(int width, int height, int nbuf,
                                          float angle_deg,
                                          ResultCallback cb, void* user_global);
DLL_EXPORT void    DLL_CALL gp_destroy_ctx(GpuCtx* ctx);

// 非ブロッキング: 空きが無ければ -2
DLL_EXPORT int     DLL_CALL gp_submit_try (GpuCtx* ctx, int frameId,
                                           const uint8_t* src, int pitchBytes,
                                           void* user_per_job);

// ブロッキング: 空きを待つ。timeout_ms<=0 なら無期限。タイムアウトで -2
DLL_EXPORT int     DLL_CALL gp_submit_wait(GpuCtx* ctx, int frameId,
                                           const uint8_t* src, int pitchBytes,
                                           void* user_per_job, int timeout_ms);
} // extern "C"

//================ 内部実装 =================//

// ---- 簡易固定スレッドプール（FFT用） ----
class ThreadPool {
public:
    explicit ThreadPool(size_t n){ start(n); }
    ~ThreadPool(){ stop(); }

    template<class F>
    void submit(F&& f){
        {
            std::lock_guard<std::mutex> lk(mu_);
            q_.emplace(std::function<void()>(std::forward<F>(f)));
        }
        cv_.notify_one();
    }

private:
    void start(size_t n){
        if (n<1) n=1;
        for(size_t i=0;i<n;++i){
            ws_.emplace_back([this]{
                for(;;){
                    std::function<void()> job;
                    {
                        std::unique_lock<std::mutex> lk(mu_);
                        cv_.wait(lk,[&]{ return stop_ || !q_.empty(); });
                        if (stop_ && q_.empty()) return;
                        job = std::move(q_.front()); q_.pop();
                    }
                    try { job(); } catch(...) { /* log等 */ }
                }
            });
        }
    }
    void stop(){
        { std::lock_guard<std::mutex> lk(mu_); stop_=true; }
        cv_.notify_all();
        for (auto& t: ws_) if (t.joinable()) t.join();
    }

    std::vector<std::thread> ws_;
    std::queue<std::function<void()>> q_;
    std::mutex mu_; std::condition_variable cv_;
    bool stop_ = false;
};

// ---- Job / Slot ----
struct Job {
    int frame_id;
    int slot;
    void* user; // per-submit の user
};

struct Slot {
    int id = -1; // -1:空き, -2:予約, >=0:frameId（使用中）

    // ホスト(Pinned) と GPU
    cv::cuda::HostMem pin_in, pin_rot;
    cv::Mat in_mat, rot_mat;          // rot_mat: 回転後 float32 1ch
    cv::cuda::GpuMat d_in, d_rot;

    // CUDA Events
    cudaEvent_t evH2D=nullptr, evK=nullptr, evD2H=nullptr;

    // FFTワーク
    cv::Mat fft_complex, fft_mag;     // CV_32FC2 / CV_32FC1
};

// ---- コンテキスト ----
struct GpuCtx {
    int W=0, H=0, N=0;
    cv::Mat rotM;
    std::vector<Slot> slots;

    // 単一のロック/条件変数（シンプル派）
    std::mutex mu;
    std::condition_variable cv;

    // スロット空き管理 & ジョブキュー
    std::queue<int> freeSlots;   // 空きslot番号
    std::queue<Job> jobQueue;    // GPUワーカー行き

    // 終了フラグ／ワーカー
    bool quitting = false;
    std::thread gpuWorker;

    // CUDA ストリーム
    cv::cuda::Stream sH2D, sK, sD2H;

    // FFT 専用スレッドプール
    std::unique_ptr<ThreadPool> fftPool;

    // コールバック
    ResultCallback cb = nullptr;
    void* user_global = nullptr;
};

// ---- ヘルパ ----
static void make_hostmat(cv::cuda::HostMem& hm, int h, int w, int type, cv::Mat& header){
    hm.release();
    hm = cv::cuda::HostMem(h, w, type, cv::cuda::HostMem::PAGE_LOCKED);
    header = hm.createMatHeader();
}

// ---- GPUワーカー ----
static void gpu_worker_loop(GpuCtx* ctx){
    auto& sH2D = ctx->sH2D;
    auto& sK   = ctx->sK;
    auto& sD2H = ctx->sD2H;

    while (true){
        Job j;
        {
            std::unique_lock<std::mutex> lk(ctx->mu);
            ctx->cv.wait(lk, [&]{ return ctx->quitting || !ctx->jobQueue.empty(); });
            if (ctx->quitting && ctx->jobQueue.empty()) break;
            j = ctx->jobQueue.front(); ctx->jobQueue.pop();
        }

        // ここからは slot を専有
        auto& sl = ctx->slots[j.slot];
        sl.id = j.frame_id; // 予約(-2) → 実使用(frameId)

        // 1) H2D
        sl.d_in.upload(sl.in_mat, sH2D);
        cudaEventRecord(sl.evH2D, cv::cuda::StreamAccessor::getStream(sH2D));

        // 2) 回転（H2Dに依存）
        sK.waitEvent(sl.evH2D);
        cv::cuda::warpAffine(sl.d_in, sl.d_rot, ctx->rotM, sl.d_rot.size(),
                             cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0), sK);
        cudaEventRecord(sl.evK, cv::cuda::StreamAccessor::getStream(sK));

        // 3) D2H（Kernelに依存）
        sD2H.waitEvent(sl.evK);
        sl.d_rot.download(sl.rot_mat, sD2H);
        cudaEventRecord(sl.evD2H, cv::cuda::StreamAccessor::getStream(sD2H));

        // D2H 完了を同期（ここで GPU は手離れ）
        cudaEventSynchronize(sl.evD2H);

        // 4) FFT は“プールへ投げる” → GPUワーカーは即次ジョブへ
        GpuCtx* ctx2 = ctx;
        int slot_idx = j.slot;
        int fid = j.frame_id;
        void* user = j.user;

        ctx->fftPool->submit([ctx2, slot_idx, fid, user]{
            auto& s = ctx2->slots[slot_idx];

            // （任意）過剰並列回避：OpenCV内部スレッドをOFF
            // cv::setNumThreads(1);

            // FFT（複素）→ magnitude
            cv::dft(s.rot_mat, s.fft_complex, cv::DFT_COMPLEX_OUTPUT);
            cv::Mat planes[2];
            cv::split(s.fft_complex, planes);
            cv::magnitude(planes[0], planes[1], s.fft_mag); // CV_32FC1

            // コールバック（最終結果のみ）
            if (ctx2->cb){
                ctx2->cb(fid,
                         reinterpret_cast<const float*>(s.fft_mag.ptr()),
                         ctx2->W, ctx2->H,
                         static_cast<int>(s.fft_mag.step),
                         user ? user : ctx2->user_global);
            }

            // スロット解放 → 空き通知
            {
                std::lock_guard<std::mutex> lk(ctx2->mu);
                s.id = -1;
                ctx2->freeSlots.push(slot_idx);
            }
            ctx2->cv.notify_all();
        });
    }
}

//================ 公開API =================//

extern "C" {

DLL_EXPORT GpuCtx* DLL_CALL gp_create_ctx(int width, int height, int nbuf,
                                          float angle_deg,
                                          ResultCallback cb, void* user_global)
{
    if (nbuf < 2) nbuf = 2;

    auto ctx = new GpuCtx();
    ctx->W = width; ctx->H = height; ctx->N = nbuf;
    ctx->cb = cb; ctx->user_global = user_global;

    // 回転行列
    cv::Point2f c(width/2.f, height/2.f);
    ctx->rotM = cv::getRotationMatrix2D(c, angle_deg, 1.0);

    // スロット確保（Pinned/GPU/Events）
    ctx->slots.resize(nbuf);
    for (int i=0;i<nbuf;++i){
        auto& s = ctx->slots[i];
        s.id = -1;
        make_hostmat(s.pin_in,  height, width, CV_8UC1,  s.in_mat);
        make_hostmat(s.pin_rot, height, width, CV_32FC1, s.rot_mat);
        s.d_in .create(height, width, CV_8UC1);
        s.d_rot.create(height, width, CV_32FC1);
        cudaEventCreateWithFlags(&s.evH2D, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&s.evK,   cudaEventDisableTiming);
        cudaEventCreateWithFlags(&s.evD2H, cudaEventDisableTiming);

        ctx->freeSlots.push(i); // 全部空き
    }

    // FFT プール起動（物理コアに合わせ調整）
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    unsigned nfft = std::max(2u, hw/2); // まずは物理コアの半分
    ctx->fftPool = std::make_unique<ThreadPool>(nfft);

    // GPUワーカー起動（1本でOK：複数ストリームで重ねる）
    ctx->gpuWorker = std::thread(gpu_worker_loop, ctx);
    return ctx;
}

DLL_EXPORT void DLL_CALL gp_destroy_ctx(GpuCtx* ctx){
    if (!ctx) return;

    // submit待ち/worker待ちを起こす
    {
        std::lock_guard<std::mutex> lk(ctx->mu);
        ctx->quitting = true;
    }
    ctx->cv.notify_all();

    if (ctx->gpuWorker.joinable()) ctx->gpuWorker.join();

    // FFTプール停止
    ctx->fftPool.reset();

    // CUDAリソース解放
    for (auto& s : ctx->slots){
        if (s.evH2D) cudaEventDestroy(s.evH2D);
        if (s.evK)   cudaEventDestroy(s.evK);
        if (s.evD2H) cudaEventDestroy(s.evD2H);
    }
    delete ctx;
}

// 非ブロッキング（空き無しなら -2）
DLL_EXPORT int DLL_CALL gp_submit_try(GpuCtx* ctx, int frameId,
                                      const uint8_t* src, int pitchBytes,
                                      void* user_per_job)
{
    if (!ctx || !src) return -1;

    int slot = -1;
    {
        std::lock_guard<std::mutex> lk(ctx->mu);
        if (ctx->freeSlots.empty()) return -2;
        slot = ctx->freeSlots.front(); ctx->freeSlots.pop();
        ctx->slots[slot].id = -2; // 予約
    }

    // 入力を Pinned に即コピー（呼び出し側寿命から解放）
    auto& s = ctx->slots[slot];
    for (int y=0; y<ctx->H; ++y){
        std::memcpy(s.in_mat.ptr(y), src + y*pitchBytes, ctx->W);
    }

    // Job をキューへ → worker 起こす
    {
        std::lock_guard<std::mutex> lk(ctx->mu);
        ctx->jobQueue.push(Job{frameId, slot, user_per_job});
    }
    ctx->cv.notify_all();
    return 0;
}

// ブロッキング（空きが出るまで待つ／タイムアウトあり）
DLL_EXPORT int DLL_CALL gp_submit_wait(GpuCtx* ctx, int frameId,
                                       const uint8_t* src, int pitchBytes,
                                       void* user_per_job, int timeout_ms)
{
    if (!ctx || !src) return -1;

    int slot = -1;
    {
        std::unique_lock<std::mutex> lk(ctx->mu);
        auto pred = [&]{ return ctx->quitting || !ctx->freeSlots.empty(); };
        if (timeout_ms <= 0) {
            ctx->cv.wait(lk, pred);
            if (ctx->quitting) return -3;
        } else {
            if (!ctx->cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), pred))
                return -2; // タイムアウト
            if (ctx->quitting) return -3;
        }
        slot = ctx->freeSlots.front(); ctx->freeSlots.pop();
        ctx->slots[slot].id = -2; // 予約
    }

    auto& s = ctx->slots[slot];
    for (int y=0; y<ctx->H; ++y){
        std::memcpy(s.in_mat.ptr(y), src + y*pitchBytes, ctx->W);
    }

    {
        std::lock_guard<std::mutex> lk(ctx->mu);
        ctx->jobQueue.push(Job{frameId, slot, user_per_job});
    }
    ctx->cv.notify_all();
    return 0;
}

} // extern "C"