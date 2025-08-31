// gpu_sync_allinone.cpp
// ビルド例(概念): cl /O2 /MD /EHsc /LD gpu_sync_allinone.cpp /I<OpenCV\include> <opencv libs...>
//                 or with CMake (単一ソースを SHARED にして OpenCV CUDA とリンク)
// 依存: OpenCV CUDA (core, imgproc, cudaimgproc, cudawarping)

#include <cstdint>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <future>
#include <vector>
#include <cstring>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/cuda.hpp>

#ifdef _WIN32
  #define DLL_EXPORT __declspec(dllexport)
  #define DLL_CALL   __cdecl
#else
  #define DLL_EXPORT
  #define DLL_CALL
#endif

extern "C" {
// 不透明ハンドル
typedef struct GpuCtx GpuCtx;

// 初期化：nbuf は 2～3 推奨。angle_deg は回転角（度）。
DLL_EXPORT GpuCtx* DLL_CALL gp_create_ctx(int width, int height, int nbuf, float angle_deg);

// 破棄
DLL_EXPORT void DLL_CALL gp_destroy_ctx(GpuCtx*);

// 同期API（公開は同期・内部は非同期）
// 入力: 8UC1 (inPitchBytes 行ピッチ)、出力: FFTのmagnitude (32FC1, outPitchBytes 行ピッチ)
DLL_EXPORT int DLL_CALL gp_process_sync(
    GpuCtx* ctx,
    const uint8_t* inPtr, int inPitchBytes,
    float* outPtr, int outPitchBytes);
} // extern "C"

//====================== 内部実装 ======================//

// --- シンプル固定スレッドプール（FFT用） ---
class ThreadPool {
public:
    explicit ThreadPool(size_t n){ start(n); }
    ~ThreadPool(){ stop(); }
    template<class F>
    void submit(F&& f){
        { std::lock_guard<std::mutex> lk(mu_); q_.emplace(std::function<void()>(std::forward<F>(f))); }
        cv_.notify_one();
    }
private:
    void start(size_t n){
        if (n < 1) n = 1;
        for(size_t i=0;i<n;++i){
            ws_.emplace_back([this]{
                for(;;){
                    std::function<void()> job;
                    { std::unique_lock<std::mutex> lk(mu_);
                      cv_.wait(lk,[&]{ return stop_ || !q_.empty(); });
                      if (stop_ && q_.empty()) return;
                      job = std::move(q_.front()); q_.pop();
                    }
                    try { job(); } catch(...) { /* ここでログしてもよい */ }
                }
            });
        }
    }
    void stop(){
        { std::lock_guard<std::mutex> lk(mu_); stop_ = true; }
        cv_.notify_all();
        for (auto& t: ws_) if (t.joinable()) t.join();
    }
    std::vector<std::thread> ws_;
    std::queue<std::function<void()>> q_;
    std::mutex mu_; std::condition_variable cv_; bool stop_ = false;
};

// --- ジョブ/スロット ---
struct Job {
    int slot = -1;
    float* outPtr = nullptr;
    int outPitch = 0;
    std::promise<void> done;
};

struct Slot {
    int id = -1; // 未使用は <0
    // Host (Pinned)
    cv::cuda::HostMem pin_in, pin_out;
    cv::Mat in_mat, out_mat;            // out_mat = 回転後 float32 1ch
    // Device
    cv::cuda::GpuMat d_in, d_out;
    // CUDA Events
    cudaEvent_t evH2D=nullptr, evK=nullptr, evD2H=nullptr;
    // 紐づくジョブ
    Job* job = nullptr;

    // FFTワークエリア（ホスト）
    cv::Mat fft_complex; // CV_32FC2
    cv::Mat fft_mag;     // CV_32FC1
};

// --- コンテキスト ---
struct GpuCtx {
    int W=0, H=0, N=0;
    cv::Mat rotM;                 // 回転行列
    std::vector<Slot> slots;

    // ストリーム分離（H2D, Kernel, D2H）
    cv::cuda::Stream sH2D, sK, sD2H;

    // ジョブキュー & ワーカー（GPU担当は常にこのスレッド）
    std::thread worker;
    std::mutex muQ; std::condition_variable cvQ;
    std::queue<Job*> qJobs;
    bool stop=false;

    // CPU FFT スレッドプール
    std::unique_ptr<ThreadPool> fftPool;
};

// --- ユーティリティ ---
static void make_hostmat(cv::cuda::HostMem& hm, int rows, int cols, int type, cv::Mat& header){
    hm.release();
    hm = cv::cuda::HostMem(rows, cols, type, cv::cuda::HostMem::PAGE_LOCKED);
    header = hm.createMatHeader(); // データ共用のMatヘッダ
}

// --- GPUワーカー（内部非同期パイプライン） ---
static void worker_loop(GpuCtx* ctx){
    auto& sH2D = ctx->sH2D; auto& sK = ctx->sK; auto& sD2H = ctx->sD2H;
    std::vector<int> inflight; inflight.reserve(ctx->N);

    while (true) {
        Job* j = nullptr;
        {
            std::unique_lock<std::mutex> lk(ctx->muQ);
            ctx->cvQ.wait(lk, [&]{ return ctx->stop || !ctx->qJobs.empty() || !inflight.empty(); });
            if (ctx->stop && ctx->qJobs.empty() && inflight.empty()) break;
            if (!ctx->qJobs.empty() && (int)inflight.size() < ctx->N) {
                j = ctx->qJobs.front(); ctx->qJobs.pop();
            }
        }

        if (j) {
            auto& s = ctx->slots[j->slot];
            s.job = j;

            // 1) H2D（非同期）
            s.d_in.upload(s.in_mat, sH2D);
            cudaEventRecord(s.evH2D, cv::cuda::StreamAccessor::getStream(sH2D));

            // 2) 回転カーネル（依存：H2D）
            sK.waitEvent(s.evH2D);
            cv::cuda::warpAffine(s.d_in, s.d_out, ctx->rotM, s.d_out.size(),
                                 cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0), sK);
            cudaEventRecord(s.evK, cv::cuda::StreamAccessor::getStream(sK));

            // 3) D2H（依存：Kernel）
            sD2H.waitEvent(s.evK);
            s.d_out.download(s.out_mat, sD2H);
            cudaEventRecord(s.evD2H, cv::cuda::StreamAccessor::getStream(sD2H));

            inflight.push_back(j->slot);
        }

        // 4) 完了したスロットだけ取り出し → FFTプールへ
        for (int i=(int)inflight.size()-1; i>=0; --i){
            int sl = inflight[i];
            auto& s = ctx->slots[sl];
            if (cudaEventQuery(s.evD2H) == cudaSuccess) {
                Job* jj = s.job;

                // FFTジョブ：回転済み out_mat → FFT(複素) → magnitude → 呼出側 outPtrへコピー → done
                ctx->fftPool->submit([&s, jj, ctx]{
                    // 複素出力 (CV_32FC2)
                    cv::dft(s.out_mat, s.fft_complex, cv::DFT_COMPLEX_OUTPUT);
                    cv::Mat planes[2];
                    cv::split(s.fft_complex, planes);
                    cv::magnitude(planes[0], planes[1], s.fft_mag); // CV_32FC1

                    // 呼び出し側バッファへコピー（行ピッチに合わせて）
                    for (int y=0; y<ctx->H; ++y){
                        std::memcpy((uint8_t*)jj->outPtr + y*jj->outPitch,
                                    s.fft_mag.ptr(y),
                                    ctx->W * sizeof(float));
                    }

                    // 完了通知 & スロット解放
                    jj->done.set_value();
                    s.id = -1; s.job = nullptr;
                });

                inflight.erase(inflight.begin()+i);
            }
        }

        // CPU張り付き防止（状況で調整）
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

//====================== 公開API実装 ======================//

extern "C" {

DLL_EXPORT GpuCtx* DLL_CALL gp_create_ctx(int width, int height, int nbuf, float angle_deg){
    if (nbuf < 2) nbuf = 2;

    auto ctx = new GpuCtx();
    ctx->W = width; ctx->H = height; ctx->N = nbuf;

    // 回転行列
    cv::Point2f c(width/2.f, height/2.f);
    ctx->rotM = cv::getRotationMatrix2D(c, angle_deg, 1.0);

    // スロット確保（Pinned/GPU/Events）
    ctx->slots.resize(nbuf);
    for (int i=0; i<nbuf; ++i){
        auto& s = ctx->slots[i];
        make_hostmat(s.pin_in,  height, width, CV_8UC1,  s.in_mat);
        make_hostmat(s.pin_out, height, width, CV_32FC1, s.out_mat);
        s.d_in.create(height, width, CV_8UC1);
        s.d_out.create(height, width, CV_32FC1);
        cudaEventCreateWithFlags(&s.evH2D, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&s.evK,   cudaEventDisableTiming);
        cudaEventCreateWithFlags(&s.evD2H, cudaEventDisableTiming);
    }

    // FFTプール起動（物理コア/メモリ帯域に合わせて調整）
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    unsigned nfft = std::max(2u, hw/2); // 目安：半分
    ctx->fftPool = std::make_unique<ThreadPool>(nfft);

    // GPUワーカー起動（CUDAは常にこのスレッドからのみ呼ぶ）
    ctx->worker = std::thread(worker_loop, ctx);
    return ctx;
}

DLL_EXPORT void DLL_CALL gp_destroy_ctx(GpuCtx* ctx){
    if (!ctx) return;

    { std::lock_guard<std::mutex> lk(ctx->muQ); ctx->stop = true; }
    ctx->cvQ.notify_all();
    if (ctx->worker.joinable()) ctx->worker.join();

    // FFTプール停止
    ctx->fftPool.reset();

    for (auto& s : ctx->slots){
        if (s.evH2D) cudaEventDestroy(s.evH2D);
        if (s.evK)   cudaEventDestroy(s.evK);
        if (s.evD2H) cudaEventDestroy(s.evD2H);
    }
    delete ctx;
}

DLL_EXPORT int DLL_CALL gp_process_sync(
    GpuCtx* ctx, const uint8_t* inPtr, int inPitch, float* outPtr, int outPitch)
{
    if (!ctx || !inPtr || !outPtr) return -1;

    // 空きスロット取得（簡易版。実運用はフリーリストを推奨）
    int slot = -1;
    for (int i=0; i<ctx->N; ++i){
        if (ctx->slots[i].id < 0) { slot = i; break; }
    }
    if (slot < 0) return -2; // in-flight 満杯

    auto& s = ctx->slots[slot];
    s.id = 1;

    // 入力を Pinned に即コピー（呼び出し側の配列寿命に依存しない）
    for (int y=0; y<ctx->H; ++y){
        std::memcpy(s.in_mat.ptr(y), inPtr + y*inPitch, ctx->W);
    }

    // ジョブ作成→GPUワーカーへ
    auto job = new Job();
    job->slot = slot; job->outPtr = outPtr; job->outPitch = outPitch;
    {
        std::lock_guard<std::mutex> lk(ctx->muQ);
        ctx->qJobs.push(job);
    }
    ctx->cvQ.notify_one();

    // 自分のジョブだけ待つ（公開APIは同期）
    job->done.get_future().wait();
    delete job;
    return 0;
}

} // extern "C"


// Program.cs（最小）
using System;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

static class Native {
  [DllImport("gpu_sync.dll", CallingConvention=CallingConvention.Cdecl)]
  public static extern IntPtr gp_create_ctx(int w,int h,int nbuf,float angle);
  [DllImport("gpu_sync.dll", CallingConvention=CallingConvention.Cdecl)]
  public static extern void gp_destroy_ctx(IntPtr ctx);
  [DllImport("gpu_sync.dll", CallingConvention=CallingConvention.Cdecl)]
  public static extern int gp_process_sync(IntPtr ctx, IntPtr inPtr,int inPitch, IntPtr outPtr,int outPitch);
}

class Program {
  static void Main(){
    int W=1024, H=768;
    IntPtr ctx = Native.gp_create_ctx(W,H,3,17f);
    var po = new ParallelOptions{ MaxDegreeOfParallelism = 2 };

    Parallel.For(0, 100, po, f=>{
      var input  = new byte[W*H];      // 8UC1
      var output = new float[W*H];     // 32FC1 (FFT magnitude が返る)
      Fill(input, W, H, f);

      unsafe {
        fixed(byte*  pIn  = input)
        fixed(float* pOut = output){
          int rc = Native.gp_process_sync(ctx, (IntPtr)pIn,  W, (IntPtr)pOut, W*sizeof(float));
          if (rc!=0) throw new Exception($"rc={rc}");
        }
      }

      // output を使って次段へ…
      if ((f%10)==0) Console.WriteLine($"frame {f} done");
    });

    Native.gp_destroy_ctx(ctx);
  }

  static void Fill(byte[] img, int w, int h, int k){
    for(int y=0;y<h;++y){ int r=y*w; for(int x=0;x<w;++x) img[r+x]=(byte)((x+y+k*3)&0xFF); }
  }
}

