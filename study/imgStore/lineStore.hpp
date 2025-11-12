#pragma once
// LineStore.hpp (C++17 / x64)
// ラインセンサー用ラインバッファ（ROI + 時刻補間）
// 想定: 1 writer(PushBlock) + N readers(TryGet...) ロックレス

#include <cstdint>
#include <vector>
#include <atomic>
#include <chrono>

namespace yourapp::imaging {

enum class PixelType : int { U8 = 0, U16 = 1 };

class LineStore {
public:
    using i64 = long long;

    // ---- 構成情報（読み取り専用） ----
    int       SourceWidth()     const noexcept;
    int       RoiX()            const noexcept;
    int       Width()           const noexcept;          // ROI 幅
    i64       CapacityLines()   const noexcept;
    int       WarmupMax()       const noexcept;
    PixelType PixelT()          const noexcept;
    int       ElemSizeBytes()   const noexcept;
    int       RowBytes()        const noexcept;          // Width * ElemSizeBytes
    int       SourceRowBytes()  const noexcept;          // SourceWidth * ElemSizeBytes

    // ---- 統計（読み取り） ----
    i64 HeadTotal()   const noexcept; // 累積取り込み行数
    i64 StoredLines() const noexcept; // 現在保持している行数

    // ---- 生成/破棄 ----
    LineStore(int srcWidth, int roiX, int roiW,
              i64 capacityLines, int warmupMax, PixelType pt);
    ~LineStore();

    // 明示破棄（多重呼び出し可）
    void Dispose() noexcept;

    // ---- コミット：ウォームアップ終了 → 線形追記へ移行 ----
    void Commit();

    // ---- PushBlock（取得ブロックの取り込み）----
    bool PushBlock(const void* src, int rows, int srcStrideBytes); // 時刻未指定→現在UTC
    bool PushBlock(const void* src, int rows, int srcStrideBytes,
                   std::chrono::system_clock::time_point acquiredUtc);
    bool PushBlock(const void* src, int rows, int srcStrideBytes,
                   double acquiredUtcSec); // UTC秒

    // ---- 窓取得（窓先頭行の時刻も返す版）----
    bool TryGetLatestWindowPtr(int winW, int winH, int x0,
                               const void*& ptr, int& strideBytes, double& timeSecAtTop) const noexcept;

    bool TryGetWindowPtr(i64 startRow, int winW, int winH, int x0,
                         const void*& ptr, int& strideBytes, double& timeSecAtTop) const noexcept;

    // ---- 互換（時刻不要版）----
    bool TryGetLatestWindowPtr(int winW, int winH, int x0,
                               const void*& ptr, int& strideBytes) const noexcept;
    bool TryGetWindowPtr(i64 startRow, int winW, int winH, int x0,
                         const void*& ptr, int& strideBytes) const noexcept;

    // コピー禁止／ムーブ可（必要なら調整）
    LineStore(const LineStore&) = delete;
    LineStore& operator=(const LineStore&) = delete;
    LineStore(LineStore&&) = delete;
    LineStore& operator=(LineStore&&) = delete;

private:
    // -------- 内部状態 --------
    int       sourceWidth_{};
    int       roiX_{};
    int       width_{};
    i64       capacityLines_{};
    int       warmupMax_{};
    PixelType pixelType_{};
    int       elemSizeBytes_{};

    std::atomic<bool> committed_{false};
    int               warmupCount_{0};     // 0..warmupMax_
    std::uint8_t*     buf_{nullptr};       // [CapacityLines x RowBytes]
    i64               writeIndex_{0};      // Commit 後に増加（起点は warmupMax_）
    std::atomic<i64>  headTotal_{0};       // 統計
    std::atomic<i64>  storedLines_{0};     // 0..warmupMax_ → warmupMax_..capacityLines_
    std::atomic<bool> disposed_{false};

    // ウォームアップ per-line 時刻
    std::vector<double> warmupTimes_;

    // Commit 後の時刻セグメント
    struct TimeSeg { i64 Start; double T; }; // Start: 論理行(Commit起点)
    std::vector<TimeSeg>      segs_;
    std::atomic<int>          segCount_{0};     // 読み手への公開は atomic
    double                    warmupLastTimeSec_{};
    i64                       commitBase_{0};   // Commit 時の既存行数 (= warmupCount_)

private:
    // ---- util ----
    static double NowUnixSec();
    static double ToUnixSec(std::chrono::system_clock::time_point tp);
    static int    clamp(int v, int lo, int hi) noexcept;
    void          check_not_disposed() const;

    // ---- セグメント管理 ----
    void AddSeg(i64 startLogical, double t);

    // ---- 行の時刻（絶対行 → ウォームアップ per-line／以降は論理行補間/外挿）----
    double RowTimeSec(i64 rowAbs) const noexcept;

    // ---- 実処理 ----
    void  PushWarmup(const void* src, int rows, int srcStrideBytes, double timeSec);
    bool  PushLinear(const void* src, int rows, int srcStrideBytes, double timeSec);
};

} // namespace yourapp::imaging
