// LineStore.cpp
#include "lineStore.hpp"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <limits>
#include <algorithm>

namespace yourapp::imaging {

// ---- 構成情報（インライン化しない版）----
int  LineStore::SourceWidth()    const noexcept { return sourceWidth_; }
int  LineStore::RoiX()           const noexcept { return roiX_; }
int  LineStore::Width()          const noexcept { return width_; }
LineStore::i64 LineStore::CapacityLines() const noexcept { return capacityLines_; }
int  LineStore::WarmupMax()      const noexcept { return warmupMax_; }
PixelType LineStore::PixelT()    const noexcept { return pixelType_; }
int  LineStore::ElemSizeBytes()  const noexcept { return elemSizeBytes_; }
int  LineStore::RowBytes()       const noexcept { return width_ * elemSizeBytes_; }
int  LineStore::SourceRowBytes() const noexcept { return sourceWidth_ * elemSizeBytes_; }

LineStore::i64 LineStore::HeadTotal()   const noexcept { return headTotal_.load(std::memory_order_relaxed); }
LineStore::i64 LineStore::StoredLines() const noexcept { return storedLines_.load(std::memory_order_acquire); }

// ---- 生成/破棄 ----
LineStore::LineStore(int srcWidth, int roiX, int roiW,
                     i64 capacityLines, int warmupMax, PixelType pt)
    : sourceWidth_(srcWidth)
    , roiX_(roiX)
    , width_(roiW)
    , capacityLines_(capacityLines)
    , warmupMax_(warmupMax)
    , pixelType_(pt)
    , elemSizeBytes_(pt == PixelType::U8 ? 1 : 2)
    , committed_(false)
    , warmupCount_(0)
    , buf_(nullptr)
    , writeIndex_(0)
    , headTotal_(0)
    , storedLines_(0)
    , disposed_(false)
    , segCount_(0)
    , warmupLastTimeSec_(std::numeric_limits<double>::quiet_NaN())
    , commitBase_(0)
{
    static_assert(sizeof(void*) == 8, "x64 専用です。");
    if (srcWidth <= 0) throw std::out_of_range("srcWidth");
    if (roiX < 0 || roiX >= srcWidth) throw std::out_of_range("roiX");
    if (roiW <= 0 || roiX + roiW > srcWidth) throw std::out_of_range("roiW");
    if (capacityLines < warmupMax || warmupMax <= 0) throw std::out_of_range("warmupMax");

    const i64 totalBytes = capacityLines_ * static_cast<i64>(RowBytes());
    if (totalBytes <= 0) throw std::overflow_error("totalBytes");
    buf_ = static_cast<std::uint8_t*>(std::malloc(static_cast<size_t>(totalBytes)));
    if (!buf_) throw std::bad_alloc();

    warmupTimes_.assign(static_cast<size_t>(warmupMax_), std::numeric_limits<double>::quiet_NaN());
    segs_.reserve(64);
}

LineStore::~LineStore() {
    Dispose();
}

void LineStore::Dispose() noexcept {
    bool expected = false;
    if (disposed_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        std::free(buf_);
        buf_ = nullptr;
    }
}

// ---- Commit ----
void LineStore::Commit() {
    check_not_disposed();
    if (committed_.load(std::memory_order_acquire)) return;

    if (std::isnan(warmupLastTimeSec_))
        warmupLastTimeSec_ = NowUnixSec();

    AddSeg(/*start(logical)=*/0, warmupLastTimeSec_);

    writeIndex_ = warmupCount_;  // 通常は warmupMax_
    commitBase_ = warmupCount_;  // 絶対行→論理行の基準

    storedLines_.store(warmupCount_, std::memory_order_release);
    committed_.store(true, std::memory_order_release);
}

// ---- PushBlock ----
bool LineStore::PushBlock(const void* src, int rows, int srcStrideBytes) {
    return PushBlock(src, rows, srcStrideBytes, NowUnixSec());
}

bool LineStore::PushBlock(const void* src, int rows, int srcStrideBytes,
                          std::chrono::system_clock::time_point acquiredUtc)
{
    const double t = ToUnixSec(acquiredUtc);
    return PushBlock(src, rows, srcStrideBytes, t);
}

bool LineStore::PushBlock(const void* src, int rows, int srcStrideBytes,
                          double acquiredUtcSec)
{
    check_not_disposed();
    if (!src) throw std::invalid_argument("src");
    if (rows <= 0) return true;
    if (srcStrideBytes < SourceRowBytes())
        throw std::invalid_argument("srcStrideBytes too small (for source width)");

    if (!committed_.load(std::memory_order_acquire)) {
        PushWarmup(src, rows, srcStrideBytes, acquiredUtcSec);
        headTotal_.fetch_add(rows, std::memory_order_relaxed);
        return true;
    } else {
        return PushLinear(src, rows, srcStrideBytes, acquiredUtcSec);
    }
}

// ---- 窓取得（時刻つき）----
bool LineStore::TryGetLatestWindowPtr(int winW, int winH, int x0,
                                      const void*& ptr, int& strideBytes, double& timeSecAtTop) const noexcept
{
    ptr = nullptr; strideBytes = RowBytes(); timeSecAtTop = std::numeric_limits<double>::quiet_NaN();
    if (winW <= 0 || winH <= 0 || winW > width_) return false;

    const i64 avail = storedLines_.load(std::memory_order_acquire);
    if (avail < winH) return false;

    const i64 topRow = avail - winH;
    return TryGetWindowPtr(topRow, winW, winH, x0, ptr, strideBytes, timeSecAtTop);
}

bool LineStore::TryGetWindowPtr(i64 startRow, int winW, int winH, int x0,
                                const void*& ptr, int& strideBytes, double& timeSecAtTop) const noexcept
{
    ptr = nullptr; strideBytes = RowBytes(); timeSecAtTop = std::numeric_limits<double>::quiet_NaN();
    if (startRow < 0 || winW <= 0 || winH <= 0 || winW > width_) return false;

    const i64 avail = storedLines_.load(std::memory_order_acquire);
    if (startRow + winH > avail) return false;

    const int x0c = clamp(x0, 0, std::max(0, width_ - winW));
    const i64 byteOffset = startRow * static_cast<i64>(RowBytes()) + static_cast<i64>(x0c) * elemSizeBytes_;
    ptr = static_cast<const void*>(buf_ + byteOffset);

    timeSecAtTop = RowTimeSec(startRow);
    return true;
}

// ---- 互換（時刻不要）----
bool LineStore::TryGetLatestWindowPtr(int winW, int winH, int x0,
                                      const void*& ptr, int& strideBytes) const noexcept
{
    double dummy;
    return TryGetLatestWindowPtr(winW, winH, x0, ptr, strideBytes, dummy);
}

bool LineStore::TryGetWindowPtr(i64 startRow, int winW, int winH, int x0,
                                const void*& ptr, int& strideBytes) const noexcept
{
    double dummy;
    return TryGetWindowPtr(startRow, winW, winH, x0, ptr, strideBytes, dummy);
}

// ---- util ----
double LineStore::NowUnixSec() {
    return ToUnixSec(std::chrono::system_clock::now());
}

double LineStore::ToUnixSec(std::chrono::system_clock::time_point tp) {
    using namespace std::chrono;
    const auto epoch = time_point<system_clock>();
    const auto dur   = duration_cast<duration<double>>(tp - epoch);
    return dur.count();
}

int LineStore::clamp(int v, int lo, int hi) noexcept {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

void LineStore::check_not_disposed() const {
    if (disposed_.load(std::memory_order_acquire)) {
        throw std::runtime_error("LineStore disposed");
    }
}

// ---- セグメント管理 ----
void LineStore::AddSeg(i64 startLogical, double t) {
    // writer 専用スレッドのみが push する前提なので vector 伸長は非同期アクセス無し
    if (segs_.size() == segs_.capacity()) {
        segs_.reserve(segs_.capacity() ? segs_.capacity() * 2 : 64);
    }
    segs_.push_back(TimeSeg{ startLogical, t });
    segCount_.store(static_cast<int>(segs_.size()), std::memory_order_release);
}

// ---- 行の時刻 ----
double LineStore::RowTimeSec(i64 rowAbs) const noexcept {
    if (!committed_.load(std::memory_order_acquire)) {
        const int idx = static_cast<int>(rowAbs);
        if (0 <= idx && idx < warmupCount_) {
            const double v = warmupTimes_[static_cast<size_t>(idx)];
            if (!std::isnan(v)) return v;
        }
        return std::isnan(warmupLastTimeSec_) ? NowUnixSec() : warmupLastTimeSec_;
    }

    if (rowAbs < commitBase_) { // コミット前（ウォームアップ領域）
        const int idx = static_cast<int>(rowAbs);
        if (0 <= idx && idx < warmupMax_) {
            const double v = warmupTimes_[static_cast<size_t>(idx)];
            if (!std::isnan(v)) return v;
        }
        return warmupLastTimeSec_;
    }

    int n = segCount_.load(std::memory_order_acquire);
    if (n == 0)
        return std::isnan(warmupLastTimeSec_) ? NowUnixSec() : warmupLastTimeSec_;

    const i64 row = rowAbs - commitBase_; // 論理行
    const auto* arr = segs_.data();

    // max(Start <= row) を二分探索
    int lo = 0, hi = n - 1, k = -1;
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        const i64 s = arr[mid].Start;
        if (s <= row) { k = mid; lo = mid + 1; }
        else          { hi = mid - 1; }
    }

    if (k < 0) return arr[0].T;
    if (k == n - 1) {
        if (n >= 2) {
            const auto& p1 = arr[n - 2]; const auto& p2 = arr[n - 1];
            const i64   dx = p2.Start - p1.Start;
            const double dt = p2.T - p1.T;
            const double a = (dx > 0) ? (dt / static_cast<double>(dx)) : 0.0; // 秒/行
            return p2.T + (row - p2.Start) * a; // 勾配外挿
        }
        return arr[n - 1].T;
    }

    const auto& prev = arr[k];
    const auto& next = arr[k + 1];
    const i64 drow = next.Start - prev.Start;
    if (drow <= 0) return prev.T;
    const double slope = (next.T - prev.T) / static_cast<double>(drow);
    return prev.T + (row - prev.Start) * slope;
}

// ---- ウォームアップ取り込み ----
void LineStore::PushWarmup(const void* src, int rows, int srcStrideBytes, double timeSec) {
    const auto* sBase = static_cast<const std::uint8_t*>(src);
    auto*       dBase = buf_;
    const int   roiByteOffset = roiX_ * elemSizeBytes_;

    int filled = warmupCount_; // 0..warmupMax_

    // 1) 未充填ぶんを埋める
    if (filled < warmupMax_) {
        const int need = warmupMax_ - filled;
        const int take = (rows < need) ? rows : need;

        for (int i = 0; i < take; ++i) {
            const auto* srcLine = sBase + static_cast<i64>(i) * srcStrideBytes + roiByteOffset;
            auto*       dstLine = dBase + static_cast<i64>(filled + i) * RowBytes();
            std::memcpy(dstLine, srcLine, static_cast<size_t>(RowBytes()));
            warmupTimes_[static_cast<size_t>(filled + i)] = timeSec; // per-line 時刻
        }

        filled += take;
        warmupCount_ = filled;
        storedLines_.store(filled, std::memory_order_release);

        rows  -= take;
        sBase += static_cast<i64>(take) * srcStrideBytes;

        if (rows <= 0) {
            warmupLastTimeSec_ = timeSec;
            return;
        }
    }

    // 2) 既に満杯：末尾 rows を反映
    if (rows >= warmupMax_) {
        const auto* tail = sBase + static_cast<i64>(rows - warmupMax_) * srcStrideBytes;
        for (int i = 0; i < warmupMax_; ++i) {
            const auto* srcLine = tail + static_cast<i64>(i) * srcStrideBytes + roiByteOffset;
            auto*       dstLine = dBase + static_cast<i64>(i) * RowBytes();
            std::memcpy(dstLine, srcLine, static_cast<size_t>(RowBytes()));
            warmupTimes_[static_cast<size_t>(i)] = timeSec;
        }
        storedLines_.store(warmupMax_, std::memory_order_release);
    }
    else { // 0 < rows < warmupMax_
        const int shift = rows;
        const int keep  = warmupMax_ - shift;

        // 前詰め（画像・時刻）
        for (int y = 0; y < keep; ++y) {
            auto*       srcRow = dBase + static_cast<i64>(y + shift) * RowBytes();
            auto*       dstRow = dBase + static_cast<i64>(y)        * RowBytes();
            std::memcpy(dstRow, srcRow, static_cast<size_t>(RowBytes()));
            warmupTimes_[static_cast<size_t>(y)] = warmupTimes_[static_cast<size_t>(y + shift)];
        }
        // 末尾 rows 行を追加（画像・時刻）
        for (int i = 0; i < rows; ++i) {
            const auto* srcLine = sBase + static_cast<i64>(i) * srcStrideBytes + roiByteOffset;
            auto*       dstLine = dBase + static_cast<i64>(keep + i) * RowBytes();
            std::memcpy(dstLine, srcLine, static_cast<size_t>(RowBytes()));
            warmupTimes_[static_cast<size_t>(keep + i)] = timeSec;
        }
        storedLines_.store(warmupMax_, std::memory_order_release);
    }

    warmupLastTimeSec_ = timeSec;
}

// ---- Commit 後：線形追記 ----
bool LineStore::PushLinear(const void* src, int rows, int srcStrideBytes, double timeSec) {
    const i64 remain = capacityLines_ - writeIndex_;
    if (remain <= 0) return false;

    const int can = static_cast<int>(std::min<i64>(remain, rows));

    // セグメント登録（論理行で）
    const i64 startAbs = writeIndex_;
    const i64 startLog = startAbs - commitBase_;
    AddSeg(startLog, timeSec);

    const auto* sBase = static_cast<const std::uint8_t*>(src);
    auto*       dBase = buf_ + writeIndex_ * RowBytes();
    const int   roiByteOffset = roiX_ * elemSizeBytes_;

    if (roiX_ == 0 && srcStrideBytes == RowBytes()) {
        const i64 bytes = static_cast<i64>(can) * RowBytes();
        std::memcpy(dBase, sBase, static_cast<size_t>(bytes));
    } else {
        for (int i = 0; i < can; ++i) {
            const auto* srcLine = sBase + static_cast<i64>(i) * srcStrideBytes + roiByteOffset;
            auto*       dstLine = dBase + static_cast<i64>(i) * RowBytes();
            std::memcpy(dstLine, srcLine, static_cast<size_t>(RowBytes()));
        }
    }

    headTotal_.fetch_add(can, std::memory_order_relaxed);
    writeIndex_ += can;

    const i64 newStored = writeIndex_;
    if (newStored > storedLines_.load(std::memory_order_relaxed))
        storedLines_.store(newStored, std::memory_order_release);

    return can == rows;
}

} // namespace yourapp::imaging
