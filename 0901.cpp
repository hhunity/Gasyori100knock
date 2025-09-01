// ==== GpuCtx に追加 ====
struct GpuCtx {
    ...
    // スロット管理（空き番号だけを保持）
    std::mutex muSlots;
    std::condition_variable cvSlots;
    std::queue<int> freeSlots;   // 空きslot番号
    bool quitting = false;       // 破棄フラグ
    ...
};

// ==== gp_create_ctx で初期化 ====
for (int i=0;i<nbuf;++i){
    auto& s = ctx->slots[i];
    s.id = -1;
    ...
    ctx->freeSlots.push(i);     // ★ 最初は全部空き
}


// ==== 完了時（workerの最後）で返却する部分 ====
// sl.id = -1; の直後に
{
    std::lock_guard<std::mutex> lk(ctx->muSlots);
    ctx->slots[j.slot].id = -1;
    ctx->freeSlots.push(j.slot);     // ★ 空きを返却
}
ctx->cvSlots.notify_one();            // ★ 待っている submit を起こす

extern "C" DLL_EXPORT int DLL_CALL
gp_submit_try(GpuCtx* ctx, int frameId, const uint8_t* src, int pitch, void* user)
{
    if (!ctx || !src) return -1;
    int slot = -1;

    // 空きを“試しに”取る（ブロックしない）
    {
        std::lock_guard<std::mutex> lk(ctx->muSlots);
        if (ctx->freeSlots.empty()) return -2;    // ★ 満杯
        slot = ctx->freeSlots.front();
        ctx->freeSlots.pop();
        ctx->slots[slot].id = -2;                 // ★ 予約マーク
    }

    // Pinned へ即コピー（寿命から解放）
    auto& s = ctx->slots[slot];
    for (int y=0; y<ctx->H; ++y)
        std::memcpy(s.in_mat.ptr(y), src + y*pitch, ctx->W);

    // Job投入
    Job j; j.frame_id = frameId; j.slot = slot; j.user = user;
    {
        std::lock_guard<std::mutex> lk(ctx->mu);
        ctx->jq.push(j);
    }
    ctx->cv.notify_one();
    return 0;
}

extern "C" DLL_EXPORT int DLL_CALL
gp_submit_wait(GpuCtx* ctx, int frameId, const uint8_t* src, int pitch, void* user, int timeout_ms)
{
    if (!ctx || !src) return -1;

    int slot = -1;
    {
        std::unique_lock<std::mutex> lk(ctx->muSlots);

        auto pred = [&]{ return ctx->quitting || !ctx->freeSlots.empty(); };

        if (timeout_ms <= 0) {
            // 無期限で待つ
            ctx->cvSlots.wait(lk, pred);
            if (ctx->quitting) return -3; // 破棄中
        } else {
            // タイムアウト付き
            if (!ctx->cvSlots.wait_for(lk, std::chrono::milliseconds(timeout_ms), pred))
                return -2; // ★ タイムアウト（空かなかった）
            if (ctx->quitting) return -3;
        }

        // 空きスロット取得＆予約
        slot = ctx->freeSlots.front();
        ctx->freeSlots.pop();
        ctx->slots[slot].id = -2;   // 予約
    }

    // 入力コピー
    auto& s = ctx->slots[slot];
    for (int y=0; y<ctx->H; ++y)
        std::memcpy(s.in_mat.ptr(y), src + y*pitch, ctx->W);

    // ジョブ投入
    Job j; j.frame_id = frameId; j.slot = slot; j.user = user;
    {
        std::lock_guard<std::mutex> lk(ctx->mu);
        ctx->jq.push(j);
    }
    ctx->cv.notify_one();
    return 0;
}

DLL_EXPORT void DLL_CALL gp_destroy_ctx(GpuCtx* ctx){
    if (!ctx) return;
    {
        // worker 側
        std::lock_guard<std::mutex> lk(ctx->mu);
        ctx->quit = true;
    }
    ctx->cv.notify_all();

    {
        // submit 側（待機中）を起こす
        std::lock_guard<std::mutex> lk(ctx->muSlots);
        ctx->quitting = true;
    }
    ctx->cvSlots.notify_all();

    if (ctx->worker.joinable()) ctx->worker.join();
    ...
}










#pragma once
#include <cstdint>

#ifdef _WIN32
  #ifdef GPUSYNC_EXPORTS
    #define GPU_API __declspec(dllexport)
  #else
    #define GPU_API __declspec(dllimport)
  #endif
  #define GPU_CALL __cdecl
#else
  #define GPU_API
  #define GPU_CALL
#endif

extern "C" {

// コンテキストハンドル
typedef struct GpuCtx GpuCtx;

// 初期化
GPU_API GpuCtx* GPU_CALL gp_create_ctx(int width, int height, int nbuf, float angle_deg);

// 破棄
GPU_API void GPU_CALL gp_destroy_ctx(GpuCtx*);

// 同期API：呼んだスレッドは自分のジョブが終わるまで待つ
// inPtr : 入力 8UC1, inPitchBytes: 入力ピッチ
// outPtr: 出力 float32, outPitchBytes: 出力ピッチ
GPU_API int GPU_CALL gp_process_sync(
    GpuCtx* ctx,
    const uint8_t* inPtr, int inPitchBytes,
    float* outPtr, int outPitchBytes);

}

#define GPUSYNC_EXPORTS
#include "gpu_sync.h"

#include <opencv2/core.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/cuda.hpp>

#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <future>

struct Job {
    int slot = -1;
    float* outPtr = nullptr;
    int outPitch = 0;
    std::promise<void> done;
};

struct Slot {
    int id = -1;
    cv::cuda::HostMem pin_in, pin_out;
    cv::Mat in_mat, out_mat;
    cv::cuda::GpuMat d_in, d_out;
    cudaEvent_t evH2D=nullptr, evK=nullptr, evD2H=nullptr;
    Job* job=nullptr;
};

struct GpuCtx {
    int W=0, H=0, N=0;
    cv::Mat rotM;
    std::vector<Slot> slots;
    cv::cuda::Stream sH2D, sK, sD2H;

    std::thread worker;
    std::mutex muQ;
    std::condition_variable cvQ;
    std::queue<Job*> qJobs;
    bool stop=false;
};

static void make_hostmat(cv::cuda::HostMem& hm, int rows, int cols, int type, cv::Mat& header){
    hm.release();
    hm = cv::cuda::HostMem(rows, cols, type, cv::cuda::HostMem::PAGE_LOCKED);
    header = hm.createMatHeader();
}

static void worker_loop(GpuCtx* ctx){
    auto& sH2D=ctx->sH2D; auto& sK=ctx->sK; auto& sD2H=ctx->sD2H;
    std::vector<int> inflight; inflight.reserve(ctx->N);

    while(true){
        Job* j=nullptr;
        {
            std::unique_lock<std::mutex> lk(ctx->muQ);
            ctx->cvQ.wait(lk, [&]{ return ctx->stop || !ctx->qJobs.empty() || !inflight.empty(); });
            if (ctx->stop && ctx->qJobs.empty() && inflight.empty()) break;
            if (!ctx->qJobs.empty() && (int)inflight.size() < ctx->N) {
                j = ctx->qJobs.front(); ctx->qJobs.pop();
            }
        }
        if (j){
            auto& s = ctx->slots[j->slot];
            s.job = j;

            // H2D
            s.d_in.upload(s.in_mat, sH2D);
            cudaEventRecord(s.evH2D, cv::cuda::StreamAccessor::getStream(sH2D));
            // Kernel
            sK.waitEvent(s.evH2D);
            cv::cuda::warpAffine(s.d_in, s.d_out, ctx->rotM, s.d_out.size(),
                                 cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0), sK);
            cudaEventRecord(s.evK, cv::cuda::StreamAccessor::getStream(sK));
            // D2H
            sD2H.waitEvent(s.evK);
            s.d_out.download(s.out_mat, sD2H);
            cudaEventRecord(s.evD2H, cv::cuda::StreamAccessor::getStream(sD2H));

            inflight.push_back(j->slot);
        }

        // 完了したものを処理
        for (int i=(int)inflight.size()-1; i>=0; --i){
            int sl = inflight[i];
            auto& s = ctx->slots[sl];
            if (cudaEventQuery(s.evD2H) == cudaSuccess){
                // 出力コピー
                for (int y=0;y<ctx->H;++y){
                    memcpy((uint8_t*)s.job->outPtr + y*s.job->outPitch,
                           s.out_mat.ptr(y),
                           ctx->W*sizeof(float));
                }
                s.job->done.set_value(); // 呼び出しスレッドを解放
                s.id = -1; s.job=nullptr;
                inflight.erase(inflight.begin()+i);
            }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

extern "C" {

GPU_API GpuCtx* GPU_CALL gp_create_ctx(int width, int height, int nbuf, float angle_deg){
    if (nbuf < 2) nbuf=2;
    auto ctx = new GpuCtx();
    ctx->W=width; ctx->H=height; ctx->N=nbuf;

    cv::Point2f c(width/2.f, height/2.f);
    ctx->rotM = cv::getRotationMatrix2D(c, angle_deg, 1.0);

    ctx->slots.resize(nbuf);
    for (int i=0;i<nbuf;++i){
        auto& s=ctx->slots[i];
        make_hostmat(s.pin_in,  height, width, CV_8UC1,  s.in_mat);
        make_hostmat(s.pin_out, height, width, CV_32FC1, s.out_mat);
        s.d_in.create(height, width, CV_8UC1);
        s.d_out.create(height, width, CV_32FC1);
        cudaEventCreateWithFlags(&s.evH2D, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&s.evK,   cudaEventDisableTiming);
        cudaEventCreateWithFlags(&s.evD2H, cudaEventDisableTiming);
    }
    ctx->worker = std::thread(worker_loop, ctx);
    return ctx;
}

GPU_API void GPU_CALL gp_destroy_ctx(GpuCtx* ctx){
    if (!ctx) return;
    {
        std::lock_guard<std::mutex> lk(ctx->muQ);
        ctx->stop=true;
    }
    ctx->cvQ.notify_all();
    if (ctx->worker.joinable()) ctx->worker.join();
    for (auto& s: ctx->slots){
        if (s.evH2D) cudaEventDestroy(s.evH2D);
        if (s.evK)   cudaEventDestroy(s.evK);
        if (s.evD2H) cudaEventDestroy(s.evD2H);
    }
    delete ctx;
}

GPU_API int GPU_CALL gp_process_sync(GpuCtx* ctx,
    const uint8_t* inPtr, int inPitch, float* outPtr, int outPitch)
{
    if (!ctx || !inPtr || !outPtr) return -1;
    int slot=-1;
    for (int i=0;i<ctx->N;++i){
        if (ctx->slots[i].id<0){ slot=i; break; }
    }
    if (slot<0) return -2;

    auto& s = ctx->slots[slot];
    s.id=1;
    for (int y=0;y<ctx->H;++y)
        memcpy(s.in_mat.ptr(y), inPtr+y*inPitch, ctx->W);

    auto job=new Job();
    job->slot=slot; job->outPtr=outPtr; job->outPitch=outPitch;
    {
        std::lock_guard<std::mutex> lk(ctx->muQ);
        ctx->qJobs.push(job);
    }
    ctx->cvQ.notify_one();

    job->done.get_future().wait(); // 同期待ち
    delete job;
    return 0;
}

} // extern "C"

using System;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

class Program {
    static class Native {
        [DllImport("gpu_sync.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr gp_create_ctx(int w, int h, int nbuf, float angle);
        [DllImport("gpu_sync.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void gp_destroy_ctx(IntPtr ctx);
        [DllImport("gpu_sync.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int gp_process_sync(IntPtr ctx,
            IntPtr inPtr, int inPitch, IntPtr outPtr, int outPitch);
    }

    static void Main() {
        int W=1024, H=768;
        IntPtr ctx = Native.gp_create_ctx(W,H,3,17.0f);

        var po = new ParallelOptions { MaxDegreeOfParallelism=2 };
        Parallel.For(0,100, po, f=>{
            byte[] input = new byte[W*H];
            float[] output = new float[W*H];
            unsafe {
                fixed(byte* pIn=input)
                fixed(float* pOut=output) {
                    int rc=Native.gp_process_sync(ctx,
                        (IntPtr)pIn, W,
                        (IntPtr)pOut, W*sizeof(float));
                    if(rc!=0) throw new Exception($"rc={rc}");
                }
            }
            // ここでCPU FFTや次段処理
            Console.WriteLine($"Frame {f} done");
        });

        Native.gp_destroy_ctx(ctx);
    }
}



##############

// gpu_sync.cpp (抜粋)
struct Job {
    int slot = -1;
    // 出力先（呼び出し元のバッファ）と通知
    float* outPtr = nullptr; int outPitch = 0;
    std::promise<void> done;
};

struct Slot {
    int id = -1;
    // Host pinned
    cv::cuda::HostMem pin_in, pin_out;
    cv::Mat in_mat, out_mat;
    // Device
    cv::cuda::GpuMat d_in, d_out;
    // Events
    cudaEvent_t evH2D=nullptr, evK=nullptr, evD2H=nullptr;
    // ジョブ紐付け
    Job* job = nullptr;
};

struct GpuCtx {
    int W=0,H=0,N=0;
    cv::Mat rotM;
    std::vector<Slot> slots;

    // streams
    cv::cuda::Stream sH2D, sK, sD2H;

    // ワーカー
    std::thread worker;
    std::mutex muQ;
    std::condition_variable cvQ;
    std::queue<Job*> qJobs;
    bool stop=false;
};

// 同期API
int gp_process_sync(GpuCtx* ctx, const uint8_t* inPtr, int inPitch, float* outPtr, int outPitch)
{
    if (!ctx || !inPtr || !outPtr) return -1;

    // 空きslotを取得（簡易：線形探索）
    int slot = -1;
    for (int i=0;i<ctx->N;++i){ if (ctx->slots[i].id < 0){ slot = i; break; } }
    if (slot < 0) return -2; // 全部in-flight（満杯）

    auto& s = ctx->slots[slot];
    s.id = 1; // 使用中マーク

    // 入力を即時コピー（呼び出し元バッファの寿命に依存しない）
    for (int y=0;y<ctx->H;++y)
        memcpy(s.in_mat.ptr(y), inPtr + y*inPitch, ctx->W);

    // ジョブを作ってキューへ
    auto job = new Job();
    job->slot = slot; job->outPtr = outPtr; job->outPitch = outPitch;

    {
        std::lock_guard<std::mutex> lk(ctx->muQ);
        ctx->qJobs.push(job);
    }
    ctx->cvQ.notify_one();

    // ここで“自分のジョブだけ”待つ（GPUはDLL内の1スレッドが処理）
    job->done.get_future().wait();
    delete job;
    return 0;
}

// ワーカースレッドのループ（概念）
void worker_loop(GpuCtx* ctx){
    auto& sH2D=ctx->sH2D; auto& sK=ctx->sK; auto& sD2H=ctx->sD2H;
    std::vector<int> inflight; inflight.reserve(ctx->N);

    while(true){
        // 1) 取れるだけ取り、ストリームに投下
        Job* j=nullptr;
        {
            std::unique_lock<std::mutex> lk(ctx->muQ);
            ctx->cvQ.wait(lk, [&]{ return ctx->stop || !ctx->qJobs.empty() || !inflight.empty(); });
            if (ctx->stop && ctx->qJobs.empty() && inflight.empty()) break;
            if (!ctx->qJobs.empty() && (int)inflight.size() < ctx->N) {
                j = ctx->qJobs.front(); ctx->qJobs.pop();
            }
        }
        if (j){
            auto& s = ctx->slots[j->slot];
            s.job = j;

            // H2D
            s.d_in.upload(s.in_mat, sH2D);
            cudaEventRecord(s.evH2D, cv::cuda::StreamAccessor::getStream(sH2D));
            // K
            sK.waitEvent(s.evH2D);
            cv::cuda::warpAffine(s.d_in, s.d_out, ctx->rotM, s.d_out.size(),
                                 cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0), sK);
            cudaEventRecord(s.evK, cv::cuda::StreamAccessor::getStream(sK));
            // D2H
            sD2H.waitEvent(s.evK);
            s.d_out.download(s.out_mat, sD2H);
            cudaEventRecord(s.evD2H, cv::cuda::StreamAccessor::getStream(sD2H));

            inflight.push_back(j->slot);
        }

        // 2) 完了したものだけ返す（複数フレームをオーバーラップ）
        for (int i=(int)inflight.size()-1; i>=0; --i){
            int sl = inflight[i];
            auto& s = ctx->slots[sl];
            if (cudaEventQuery(s.evD2H) == cudaSuccess){
                // 呼び出し側バッファへコピー（float）
                for (int y=0;y<ctx->H;++y)
                    memcpy((uint8_t*)s.job->outPtr + y*s.job->outPitch, s.out_mat.ptr(y), ctx->W*sizeof(float));
                s.job->done.set_value();  // 同期呼び出しを起こしたスレッドを解放
                s.id = -1; s.job = nullptr;
                inflight.erase(inflight.begin()+i);
            }
        }
        // ほんの少しゆっくり（CPU100%回避）
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}


############

// gpu_async.h
#pragma once
#include <cstdint>

#ifdef _WIN32
  #ifdef GPUASYNC_EXPORTS
    #define GPU_API __declspec(dllexport)
  #else
    #define GPU_API __declspec(dllimport)
  #endif
  #define GPU_CALL __cdecl
#else
  #define GPU_API
  #define GPU_CALL
#endif

extern "C" {

// 返却データは float32 1ch 画像（回転済み）
// コールバックは D2H 完了直後に呼ばれる（スレッドはDLL内部のワーカー）
typedef void (GPU_CALL *ResultCallback)(
    int frame_id,          // ユーザーが submit 時に渡したID
    const float* data,     // 行先頭ポインタ（Pinned上）
    int width, int height, // 画像サイズ
    int stride_bytes,      // 行ストライド
    void* user_state       // ユーザーデータ透過
);

typedef struct GpuCtx GpuCtx;

// 初期化：nbuf は 2〜3 推奨（ダブル/トリプルバッファ）
GPU_API GpuCtx* GPU_CALL gp_create_ctx(
    int width, int height, int nbuf,
    float angle_deg,                // 回転角
    ResultCallback cb, void* user_state);

// 破棄
GPU_API void GPU_CALL gp_destroy_ctx(GpuCtx*);

// 非同期投入：host_ptr は 8bitグレー（width×height, pitchBytes 行ピッチ）
// 戻り値：0=OK, 非0=エラー
GPU_API int GPU_CALL gp_submit(
    GpuCtx*, int frame_id,
    const uint8_t* host_ptr, int pitchBytes);
}

// gpu_async.cpp
#define GPUASYNC_EXPORTS
#include "gpu_async.h"

#include <opencv2/core.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/cuda.hpp>

#include <thread>
#include <atomic>
#include <vector>
#include <queue>
#include <condition_variable>

// ---- 内部構造 ----
struct FrameSlot {
    int id = -1;

    // Host (Pinned)
    cv::cuda::HostMem pin_in;
    cv::cuda::HostMem pin_out;
    cv::Mat in_mat;   // pin_inヘッダ (CV_8UC1)
    cv::Mat out_mat;  // pin_outヘッダ (CV_32FC1)

    // Device
    cv::cuda::GpuMat d_in, d_out;

    // Events
    cudaEvent_t evH2D = nullptr;
    cudaEvent_t evK   = nullptr;
    cudaEvent_t evD2H = nullptr;
};

struct Job {
    int slot = -1;
    int frame_id = -1;
    const uint8_t* src = nullptr;
    int pitch = 0;
};

class JobQueue {
public:
    void push(Job j){ { std::lock_guard<std::mutex> lk(mu_); q_.push(std::move(j)); } cv_.notify_one(); }
    bool pop(Job& j){
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&]{ return stop_ || !q_.empty(); });
        if (q_.empty()) return false;
        j = std::move(q_.front()); q_.pop(); return true;
    }
    void stop(){ { std::lock_guard<std::mutex> lk(mu_); stop_ = true; } cv_.notify_all(); }
private:
    std::mutex mu_; std::condition_variable cv_;
    std::queue<Job> q_;
    bool stop_ = false;
};

struct GpuCtx {
    int W=0, H=0, N=0;
    ResultCallback cb=nullptr;
    void* user=nullptr;
    cv::Mat rotM;
    std::vector<FrameSlot> slots;

    // Streams
    cv::cuda::Stream sH2D, sK, sD2H;

    // ワーカー
    std::thread worker;
    JobQueue jq;
    std::atomic<bool> quit{false};
};

// ---- ユーティリティ ----
static void make_hostmat(cv::cuda::HostMem& hm, int rows, int cols, int type, cv::Mat& header){
    hm.release();
    hm = cv::cuda::HostMem(rows, cols, type, cv::cuda::HostMem::PAGE_LOCKED);
    header = hm.createMatHeader();
}

extern "C" {

GPU_API GpuCtx* GPU_CALL gp_create_ctx(
    int width, int height, int nbuf, float angle_deg,
    ResultCallback cb, void* user_state)
{
    if (nbuf < 2) nbuf = 2;
    auto ctx = new GpuCtx();
    ctx->W = width; ctx->H = height; ctx->N = nbuf;
    ctx->cb = cb;   ctx->user = user_state;

    // 回転行列
    cv::Point2f c(width/2.f, height/2.f);
    ctx->rotM = cv::getRotationMatrix2D(c, angle_deg, 1.0);

    // スロット確保（Pinned/GPU/Events）
    ctx->slots.resize(nbuf);
    for (int i=0;i<nbuf;++i){
        auto& s = ctx->slots[i];
        make_hostmat(s.pin_in,  height, width, CV_8UC1,  s.in_mat);
        make_hostmat(s.pin_out, height, width, CV_32FC1, s.out_mat);
        s.d_in.create(height, width, CV_8UC1);
        s.d_out.create(height, width, CV_32FC1);
        cudaEventCreateWithFlags(&s.evH2D, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&s.evK,   cudaEventDisableTiming);
        cudaEventCreateWithFlags(&s.evD2H, cudaEventDisableTiming);
    }

    // GPUワーカー（1本）：Submitされたジョブを順次流し、完了でcallback発火
    ctx->worker = std::thread([ctx]{
        auto& sH2D = ctx->sH2D; auto& sK = ctx->sK; auto& sD2H = ctx->sD2H;

        while (!ctx->quit.load(std::memory_order_relaxed)) {
            Job j;
            if (!ctx->jq.pop(j)) break; // stop
            if (j.slot < 0) continue;
            auto& sl = ctx->slots[j.slot];
            sl.id = j.frame_id;

            // 1) Host書き込み（Pinnedバッファへコピー or 直接受け取りでも可）
            //    ※ここでは単純コピー。実機はカメラSDKから直接Pinnedへが理想
            //    in_mat.step は行ピッチ（bytes）
            for (int y=0; y<ctx->H; ++y){
                memcpy(sl.in_mat.ptr(y), j.src + y*j.pitch, ctx->W);
            }

            // 2) H2D
            sl.d_in.upload(sl.in_mat, sH2D);
            cudaEventRecord(sl.evH2D, cv::cuda::StreamAccessor::getStream(sH2D));

            // 3) Kernel（H2D完了に同期）
            sK.waitEvent(sl.evH2D);
            cv::cuda::warpAffine(sl.d_in, sl.d_out, ctx->rotM, sl.d_out.size(),
                                 cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0), sK);
            cudaEventRecord(sl.evK, cv::cuda::StreamAccessor::getStream(sK));

            // 4) D2H（Kernel完了に同期）
            sD2H.waitEvent(sl.evK);
            sl.d_out.download(sl.out_mat, sD2H);
            cudaEventRecord(sl.evD2H, cv::cuda::StreamAccessor::getStream(sD2H));

            // 5) 完了待ち最小化：ここでは確実性のため同期。実戦は watcher で query→callback でも良い
            cudaEventSynchronize(sl.evD2H);

            // 6) コールバック通知（回転済み float32 の行先頭ポインタ/stride付き）
            if (ctx->cb){
                ctx->cb(sl.id,
                        reinterpret_cast<const float*>(sl.out_mat.ptr()),
                        ctx->W, ctx->H,
                        static_cast<int>(sl.out_mat.step),
                        ctx->user);
            }

            sl.id = -1; // 空き
        }
    });

    return ctx;
}

GPU_API void GPU_CALL gp_destroy_ctx(GpuCtx* ctx){
    if (!ctx) return;
    ctx->quit = true;
    ctx->jq.stop();
    if (ctx->worker.joinable()) ctx->worker.join();

    for (auto& s : ctx->slots){
        if (s.evH2D) cudaEventDestroy(s.evH2D);
        if (s.evK)   cudaEventDestroy(s.evK);
        if (s.evD2H) cudaEventDestroy(s.evD2H);
    }
    delete ctx;
}

GPU_API int GPU_CALL gp_submit(
    GpuCtx* ctx, int frame_id,
    const uint8_t* host_ptr, int pitchBytes)
{
    if (!ctx || !host_ptr) return -1;
    // 簡易：ラウンドロビンで空きスロットを探す（実戦はフリーリスト管理推奨）
    int slot = -1;
    for (int i=0;i<ctx->N;++i){
        if (ctx->slots[i].id < 0){ slot = i; break; }
    }
    if (slot < 0) return -2; // いっぱい（呼び出し側でリトライ/バックプレッシャ）

    Job j; j.slot = slot; j.frame_id = frame_id; j.src = host_ptr; j.pitch = pitchBytes;
    ctx->jq.push(std::move(j));
    return 0;
}

} // extern "C"

// GpuWrapper.cs
using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

internal static class Native {
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void ResultCallback(int frameId, IntPtr data, int width, int height, int stride, IntPtr user);

    [DllImport("gpu_async", CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr gp_create_ctx(int w, int h, int nbuf, float angleDeg, ResultCallback cb, IntPtr user);

    [DllImport("gpu_async", CallingConvention = CallingConvention.Cdecl)]
    public static extern void gp_destroy_ctx(IntPtr ctx);

    [DllImport("gpu_async", CallingConvention = CallingConvention.Cdecl)]
    public static extern int gp_submit(IntPtr ctx, int frameId, IntPtr hostPtr, int pitchBytes);
}

public sealed class GpuWrapper : IDisposable {
    private readonly IntPtr _ctx;
    private readonly Native.ResultCallback _cb; // GC防止
    private readonly ConcurrentDictionary<int, TaskCompletionSource<Result>> _map = new();

    public record Result(int FrameId, int Width, int Height, int StrideBytes, IntPtr DataPtr);

    public GpuWrapper(int w, int h, int nbuf, float angleDeg){
        _cb = OnResult;
        _ctx = Native.gp_create_ctx(w, h, nbuf, angleDeg, _cb, IntPtr.Zero);
        if (_ctx == IntPtr.Zero) throw new InvalidOperationException("gp_create_ctx failed");
    }

    public void Dispose(){
        Native.gp_destroy_ctx(_ctx);
    }

    public unsafe Task<Result> EnqueueAsync(byte[] frame8u, int pitchBytes, int frameId){
        var tcs = new TaskCompletionSource<Result>(TaskCreationOptions.RunContinuationsAsynchronously);
        if (!_map.TryAdd(frameId, tcs))
            throw new InvalidOperationException("duplicated frameId");
        fixed (byte* p = frame8u){
            int rc = Native.gp_submit(_ctx, frameId, (IntPtr)p, pitchBytes);
            if (rc != 0){
                _map.TryRemove(frameId, out _);
                tcs.SetException(new Exception($"gp_submit failed rc={rc}"));
            }
        }
        return tcs.Task;
    }

    // コールバックは DLL のワーカースレッドから呼ばれる
    private void OnResult(int frameId, IntPtr data, int w, int h, int stride, IntPtr user){
        if (_map.TryRemove(frameId, out var tcs)){
            tcs.SetResult(new Result(frameId, w, h, stride, data));
        }
    }
}

// 例: 回転後の結果を受け取り、CPU側スレッドプールでFFT（Task.Run）を並列化
var gpu = new GpuWrapper(width:1024, height:768, nbuf:3, angleDeg:17f);

for (int f=0; f<1000; ++f){
    byte[] frame = AcquireFrameIntoByteArraySomehow(); // 8UC1 (pitch=width)
    var task = gpu.EnqueueAsync(frame, pitchBytes:1024, frameId:f);

    _ = task.ContinueWith(t => {
        var res = t.Result;
        // res.DataPtr は回転後 float32 の先頭。必要なら managed コピーしてFFTへ。
        // ここでは例として Task.Run で並列FFTに
        return Task.Run(() => CpuFft(res));
    }).Unwrap();
}

