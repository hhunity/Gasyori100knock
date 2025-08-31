// 最終結果（回転 + FFT magnitude）をコールバックで返す非同期Submit型

extern "C" {
    typedef void (DLL_CALL *ResultCallback)(int frameId,
                                            const float* data,
                                            int width, int height,
                                            int stride,
                                            void* user);

    struct GpuCtx;
    DLL_EXPORT GpuCtx* DLL_CALL gp_create_ctx(int w, int h, int nbuf,
                                              float angle_deg,
                                              ResultCallback cb,
                                              void* user);
    DLL_EXPORT void   DLL_CALL gp_destroy_ctx(GpuCtx* ctx);
    DLL_EXPORT int    DLL_CALL gp_submit(GpuCtx* ctx, int frameId,
                                         const uint8_t* src, int pitch);
}

struct Job {
    int frame_id;
    int slot;
    const uint8_t* src;
    int pitch;
};

struct Slot {
    int id=-1;
    cv::cuda::HostMem pin_in, pin_out;
    cv::Mat in_mat, out_mat;
    cv::cuda::GpuMat d_in, d_out;
    cudaEvent_t evH2D=nullptr, evK=nullptr, evD2H=nullptr;
    cv::Mat fft_complex, fft_mag;
};

struct GpuCtx {
    int W=0, H=0, N=0;
    cv::Mat rotM;
    std::vector<Slot> slots;
    cv::cuda::Stream sH2D,sK,sD2H;

    std::thread worker;
    std::atomic<bool> quit{false};
    std::queue<Job> jq;
    std::mutex mu;
    std::condition_variable cv;

    ResultCallback cb=nullptr;
    void* user=nullptr;
};

static void make_hostmat(cv::cuda::HostMem& hm,int h,int w,int type,cv::Mat& header){
    hm.release();
    hm=cv::cuda::HostMem(h,w,type,cv::cuda::HostMem::PAGE_LOCKED);
    header=hm.createMatHeader();
}

extern "C" {

DLL_EXPORT GpuCtx* DLL_CALL gp_create_ctx(int w,int h,int nbuf,float angle_deg,
                                          ResultCallback cb, void* user){
    if (nbuf<2) nbuf=2;
    auto ctx=new GpuCtx();
    ctx->W=w; ctx->H=h; ctx->N=nbuf; ctx->cb=cb; ctx->user=user;

    cv::Point2f c(w/2.f,h/2.f);
    ctx->rotM=cv::getRotationMatrix2D(c,angle_deg,1.0);

    ctx->slots.resize(nbuf);
    for(int i=0;i<nbuf;++i){
        auto& s=ctx->slots[i];
        make_hostmat(s.pin_in,h,w,CV_8UC1,s.in_mat);
        make_hostmat(s.pin_out,h,w,CV_32FC1,s.out_mat);
        s.d_in.create(h,w,CV_8UC1);
        s.d_out.create(h,w,CV_32FC1);
        cudaEventCreateWithFlags(&s.evH2D,cudaEventDisableTiming);
        cudaEventCreateWithFlags(&s.evK,cudaEventDisableTiming);
        cudaEventCreateWithFlags(&s.evD2H,cudaEventDisableTiming);
    }

    // ワーカースレッド：sleepループではなく「キューが来たら処理」
    ctx->worker=std::thread([ctx]{
        auto& sH2D=ctx->sH2D; auto& sK=ctx->sK; auto& sD2H=ctx->sD2H;
        while(true){
            Job j;
            {
                std::unique_lock<std::mutex> lk(ctx->mu);
                ctx->cv.wait(lk,[&]{ return ctx->quit || !ctx->jq.empty(); });
                if(ctx->quit && ctx->jq.empty()) break;
                j=ctx->jq.front(); ctx->jq.pop();
            }
            auto& sl=ctx->slots[j.slot];
            sl.id=j.frame_id;

            // 入力コピー → H2D
            for(int y=0;y<ctx->H;++y)
                memcpy(sl.in_mat.ptr(y), j.src + y*j.pitch, ctx->W);
            sl.d_in.upload(sl.in_mat, sH2D);
            cudaEventRecord(sl.evH2D, cv::cuda::StreamAccessor::getStream(sH2D));

            // 回転
            sK.waitEvent(sl.evH2D);
            cv::cuda::warpAffine(sl.d_in, sl.d_out, ctx->rotM, sl.d_out.size(),
                                 cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                                 cv::Scalar(0), sK);
            cudaEventRecord(sl.evK, cv::cuda::StreamAccessor::getStream(sK));

            // D2H
            sD2H.waitEvent(sl.evK);
            sl.d_out.download(sl.out_mat, sD2H);
            cudaEventRecord(sl.evD2H, cv::cuda::StreamAccessor::getStream(sD2H));
            cudaEventSynchronize(sl.evD2H); // 完了を同期

            // FFT（CPU側）
            cv::dft(sl.out_mat, sl.fft_complex, cv::DFT_COMPLEX_OUTPUT);
            cv::Mat planes[2]; cv::split(sl.fft_complex, planes);
            cv::magnitude(planes[0], planes[1], sl.fft_mag);

            // コールバック（最終結果だけ）
            if(ctx->cb){
                ctx->cb(sl.id,
                        reinterpret_cast<const float*>(sl.fft_mag.ptr()),
                        ctx->W, ctx->H,
                        static_cast<int>(sl.fft_mag.step),
                        ctx->user);
            }

            sl.id=-1;
        }
    });

    return ctx;
}

DLL_EXPORT void DLL_CALL gp_destroy_ctx(GpuCtx* ctx){
    if(!ctx) return;
    {
        std::lock_guard<std::mutex> lk(ctx->mu);
        ctx->quit=true;
    }
    ctx->cv.notify_all();
    if(ctx->worker.joinable()) ctx->worker.join();
    for(auto& s:ctx->slots){
        if(s.evH2D) cudaEventDestroy(s.evH2D);
        if(s.evK)   cudaEventDestroy(s.evK);
        if(s.evD2H) cudaEventDestroy(s.evD2H);
    }
    delete ctx;
}

DLL_EXPORT int DLL_CALL gp_submit(GpuCtx* ctx,int frameId,const uint8_t* src,int pitch){
    if(!ctx||!src) return -1;
    int slot=-1;
    for(int i=0;i<ctx->N;++i){ if(ctx->slots[i].id<0){ slot=i; break; } }
    if(slot<0) return -2; // 満杯
    Job j{frameId,slot,src,pitch};
    {
        std::lock_guard<std::mutex> lk(ctx->mu);
        ctx->jq.push(j);
    }
    ctx->cv.notify_one();
    return 0;
}

} // extern "C"