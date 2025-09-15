// LineStore.cs  (x64 / unsafe 必須)
using System;
using System.Runtime.InteropServices;
using System.Threading;

namespace YourApp.Imaging
{
    public enum PixelType { U8 = 0, U16 = 1 }

    /// <summary>
    /// ラインセンサー用ラインバッファ（ROI + 時刻補間）
    /// - 受信元 srcWidth のうち [roiX, roiX+roiW) だけを保持（以降の Width は roiW）
    /// - ウォームアップ: 最新 WarmupMax 行を 0..WarmupMax-1 に連続配置
    ///   * ウォームアップ行は「行ごとの UTC秒(double)」を保持
    /// - Commit(): 以降 WarmupMax の続きから線形追記
    ///   * 各 PushBlock の「ブロック先頭(論理行)と時刻」をセグメントに記録
    ///   * 時刻計算は “窓先頭の絶対行” を内部で論理行に換算して線形補間（直後がなければ外挿）
    /// - 並行: 1 writer(PushBlock) + N readers(TryGet...) 前提（ロックなし公開）
    /// </summary>
    public unsafe sealed class LineStore : IDisposable
    {
        // 構成
        public int  SourceWidth    { get; }
        public int  RoiX           { get; }
        public int  Width          { get; }     // ROI 幅
        public long CapacityLines  { get; }
        public int  WarmupMax      { get; }
        public PixelType PixelType { get; }
        public int  ElemSizeBytes  { get; }
        public int  RowBytes       => Width * ElemSizeBytes;
        public int  SourceRowBytes => SourceWidth * ElemSizeBytes;

        // 状態
        private volatile bool _committed;
        private int _warmupCount;          // 0..WarmupMax
        private IntPtr _buf;               // [CapacityLines x RowBytes]
        private long _writeIndex;          // Commit後に増える（WarmupMax から）
        private long _headTotal;           // 統計
        private long _storedLines;         // 0..WarmupMax → WarmupMax..CapacityLines
        private volatile bool _disposed;

        // 時刻（ウォームアップ per-line）
        private double[] _warmupTimes;     // 長さ WarmupMax

        // 時刻（Commit後のセグメント）
        private struct TimeSeg { public long Start; public double T; } // Start: 論理行(Commit起点)
        private TimeSeg[] _segs = new TimeSeg[64];
        private int _segCount = 0;         // Volatile.Write で公開
        private double _warmupLastTimeSec = double.NaN; // ウォームアップ最後のブロック時刻
        private long _commitBase;          // Commit時点の既存行数（= WarmupMax）

        // ---- ctor / dtor ----
        /// <param name="srcWidth">受信元の横幅（例 2048）</param>
        /// <param name="roiX">保存する左端x（0..srcWidth-1）</param>
        /// <param name="roiW">保存する幅（1..srcWidth-roiX）→ 以降の Width になります</param>
        /// <param name="capacityLines">保持できる最大行数（>= warmupMax）</param>
        /// <param name="warmupMax">ウォームアップ行数（例: 6）</param>
        public LineStore(int srcWidth, int roiX, int roiW,
                         long capacityLines, int warmupMax,
                         PixelType pt)
        {
            if (IntPtr.Size == 4) throw new PlatformNotSupportedException("x64 専用です。");
            if (srcWidth <= 0) throw new ArgumentOutOfRangeException(nameof(srcWidth));
            if (roiX < 0 || roiX >= srcWidth) throw new ArgumentOutOfRangeException(nameof(roiX));
            if (roiW <= 0 || roiX + roiW > srcWidth) throw new ArgumentOutOfRangeException(nameof(roiW));
            if (capacityLines < warmupMax || warmupMax <= 0) throw new ArgumentOutOfRangeException(nameof(warmupMax));

            SourceWidth = srcWidth;
            RoiX = roiX;
            Width = roiW;
            CapacityLines = capacityLines;
            WarmupMax = warmupMax;
            PixelType = pt;
            ElemSizeBytes = (pt == PixelType.U8) ? 1 : 2;

            checked
            {
                long totalBytes = CapacityLines * (long)RowBytes;
                _buf = Marshal.AllocHGlobal((nint)totalBytes);
            }

            _warmupTimes = new double[WarmupMax];
            for (int i = 0; i < WarmupMax; i++) _warmupTimes[i] = double.NaN;

            _committed = false;
            _warmupCount = 0;
            _writeIndex = 0;
            _commitBase = 0;

            Volatile.Write(ref _storedLines, 0);
            Interlocked.Exchange(ref _headTotal, 0);
            Volatile.Write(ref _disposed, false);
        }

        public void Dispose()
        {
            if (Volatile.Read(ref _disposed)) return;
            Marshal.FreeHGlobal(_buf);
            _buf = IntPtr.Zero;
            Volatile.Write(ref _disposed, true);
        }
        /// <summary>
        /// ウォームアップ終了。現在の WarmupMax 行の続きから線形追記に移行。
        /// start=0 の基準時刻セグメントを 1 つ作成。
        /// </summary>
        public void Commit()
        {
            if (Volatile.Read(ref _disposed)) throw new ObjectDisposedException(nameof(LineStore));
            if (Volatile.Read(ref _committed)) return;

            if (double.IsNaN(_warmupLastTimeSec))
                _warmupLastTimeSec = DateTimeToUnixSec(DateTime.UtcNow);

            // start=0 (論理) の基準点を追加
            AddSeg(0, _warmupLastTimeSec);

            _writeIndex = _warmupCount;  // 通常 WarmupMax
            _commitBase = _warmupCount;  // 論理0の基準（絶対→論理変換に使用）

            Volatile.Write(ref _storedLines, _warmupCount);
            Volatile.Write(ref _committed, true);
        }

        // ---- PushBlock（取得時刻つき）----
        // 互換：時刻未指定→現在UTC
        public bool PushBlock(IntPtr src, int rows, int srcStrideBytes)
            => PushBlock(src, rows, srcStrideBytes, DateTimeToUnixSec(DateTime.UtcNow));

        // PushBlock（DateTime）
        public bool PushBlock(IntPtr src, int rows, int srcStrideBytes, DateTime acquiredUtc)
            => PushBlock(src, rows, srcStrideBytes, DateTimeToUnixSec(acquiredUtc.ToUniversalTime()));

          /// <summary>
        /// 受信ブロックを取り込み。src はソース幅 SourceWidth の画素行、stride はそのバイトピッチ。
        /// 保持するのは ROI [RoiX..RoiX+Width) のみ。
        /// 取得時刻はこのブロック先頭行の時刻（UTC秒, double）。
        /// </summary>
        public bool PushBlock(IntPtr src, int rows, int srcStrideBytes, double acquiredUtcSec)
        {
            if (Volatile.Read(ref _disposed)) throw new ObjectDisposedException(nameof(LineStore));
            if (src == IntPtr.Zero) throw new ArgumentNullException(nameof(src));
            if (rows <= 0) return true;
            if (srcStrideBytes < SourceRowBytes) throw new ArgumentException("srcStrideBytes too small (for source width)");

            if (!Volatile.Read(ref _committed))
            {
                PushWarmup(src, rows, srcStrideBytes, acquiredUtcSec);
                Interlocked.Add(ref _headTotal, rows);
                return true;
            }
            else
            {
                return PushLinear(src, rows, srcStrideBytes, acquiredUtcSec);
            }
        }

        // ウォームアップ
        private void PushWarmup(IntPtr src, int rows, int srcStrideBytes, double timeSec)
        {
            byte* sBase = (byte*)src;
            byte* dBase = (byte*)_buf;
            int roiByteOffset = RoiX * ElemSizeBytes;

            int filled = _warmupCount; // 0..WarmupMax

            // 1) 満杯まで埋める
            if (filled < WarmupMax)
            {
                int need = WarmupMax - filled;
                int take = (rows < need) ? rows : need;

                for (int i = 0; i < take; i++)
                {
                    byte* srcLine = sBase + (long)i * srcStrideBytes + roiByteOffset;
                    byte* dstLine = dBase + (long)(filled + i) * RowBytes;
                    Buffer.MemoryCopy(srcLine, dstLine, RowBytes, RowBytes);

                    _warmupTimes[filled + i] = timeSec; // per-line 時刻
                }

                filled += take;
                _warmupCount = filled;
                Volatile.Write(ref _storedLines, filled);

                rows  -= take;
                sBase += (long)take * srcStrideBytes;

                if (rows <= 0)
                {
                    _warmupLastTimeSec = timeSec;
                    return;
                }
            }

            // 2) 満杯状態の更新
            if (rows >= WarmupMax)
            {
                byte* tail = sBase + (long)(rows - WarmupMax) * srcStrideBytes;
                for (int i = 0; i < WarmupMax; i++)
                {
                    byte* srcLine = tail + (long)i * srcStrideBytes + roiByteOffset;
                    byte* dstLine = dBase + (long)i * RowBytes;
                    Buffer.MemoryCopy(srcLine, dstLine, RowBytes, RowBytes);
                    _warmupTimes[i] = timeSec;
                }
                Volatile.Write(ref _storedLines, WarmupMax);
            }
            else // 0 < rows < WarmupMax
            {
                int shift = rows;
                int keep  = WarmupMax - shift;

                // 前詰め（画像・時刻）
                for (int y = 0; y < keep; y++)
                {
                    byte* srcRow = dBase + (long)(y + shift) * RowBytes;
                    byte* dstRow = dBase + (long)y * RowBytes;
                    Buffer.MemoryCopy(srcRow, dstRow, RowBytes, RowBytes);

                    _warmupTimes[y] = _warmupTimes[y + shift];
                }

                // 末尾に rows 行追加（画像・時刻）
                for (int i = 0; i < rows; i++)
                {
                    byte* srcLine = sBase + (long)i * srcStrideBytes + roiByteOffset;
                    byte* dstLine = dBase + (long)(keep + i) * RowBytes;
                    Buffer.MemoryCopy(srcLine, dstLine, RowBytes, RowBytes);

                    _warmupTimes[keep + i] = timeSec;
                }

                Volatile.Write(ref _storedLines, WarmupMax);
            }

            _warmupLastTimeSec = timeSec;
        }

        // Commit後：線形追記
        private bool PushLinear(IntPtr src, int rows, int srcStrideBytes, double timeSec)
        {
            long remain = CapacityLines - _writeIndex;
            if (remain <= 0) return false;

            int can = (int)Math.Min(remain, rows);

            // セグメント（論理行で登録）
            long startAbs = _writeIndex;
            long startLog = startAbs - _commitBase;
            AddSeg(startLog, timeSec);

            byte* sBase = (byte*)src;
            byte* dBase = (byte*)_buf + _writeIndex * RowBytes;
            int roiByteOffset = RoiX * ElemSizeBytes;

            if (RoiX == 0 && srcStrideBytes == RowBytes)
            {
                long bytes = (long)can * RowBytes;
                Buffer.MemoryCopy(sBase, dBase, bytes, bytes);
            }
            else
            {
                for (int i = 0; i < can; i++)
                {
                    byte* srcLine = sBase + (long)i * srcStrideBytes + roiByteOffset;
                    byte* dstLine = dBase + (long)i * RowBytes;
                    Buffer.MemoryCopy(srcLine, dstLine, RowBytes, RowBytes);
                }
            }

            Interlocked.Add(ref _headTotal, can);
            _writeIndex += can;

            long newStored = _writeIndex;
            if (newStored > Volatile.Read(ref _storedLines))
                Volatile.Write(ref _storedLines, newStored);

            return can == rows;
        }

        // セグメント管理
        private void AddSeg(long start, double t)
        {
            int n = _segCount;
            if (n == _segs.Length)
            {
                var bigger = new TimeSeg[_segs.Length * 2];
                Array.Copy(_segs, bigger, _segs.Length);
                _segs = bigger;
            }
            _segs[n].Start = start;
            _segs[n].T     = t;
            Volatile.Write(ref _segCount, n + 1);
        }

        // 行の時刻（絶対行を受け、ウォームアップは per-line、以降は論理行補間/外挿）
        private double RowTimeSec(long rowAbs)
        {
            if (!Volatile.Read(ref _committed))
            {
                int idx = (int)rowAbs;
                if (idx >= 0 && idx < _warmupCount && !double.IsNaN(_warmupTimes[idx]))
                    return _warmupTimes[idx];
                return double.IsNaN(_warmupLastTimeSec) ? DateTimeToUnixSec(DateTime.UtcNow) : _warmupLastTimeSec;
            }

            if (rowAbs < _commitBase)
            {
                int idx = (int)rowAbs;
                if (idx >= 0 && idx < WarmupMax && !double.IsNaN(_warmupTimes[idx]))
                    return _warmupTimes[idx];
                return _warmupLastTimeSec;
            }

            int n = Volatile.Read(ref _segCount);
            if (n == 0)
                return double.IsNaN(_warmupLastTimeSec) ? DateTimeToUnixSec(DateTime.UtcNow) : _warmupLastTimeSec;

            long row = rowAbs - _commitBase; // 論理行
            var arr = _segs;

            // max(Start <= row) を二分探索
            int lo = 0, hi = n - 1, k = -1;
            while (lo <= hi)
            {
                int mid = (lo + hi) >> 1;
                long s = arr[mid].Start;
                if (s <= row) { k = mid; lo = mid + 1; }
                else          { hi = mid - 1; }
            }

            if (k < 0) return arr[0].T;
            if (k == n - 1)
            {
                if (n >= 2)
                {
                    var p1 = arr[n - 2]; var p2 = arr[n - 1];
                    long   dx = p2.Start - p1.Start;
                    double dt = p2.T     - p1.T;
                    double a  = (dx > 0) ? dt / dx : 0.0; // 秒/行
                    return p2.T + (row - p2.Start) * a;  // 勾配外挿
                }
                return arr[n - 1].T;
            }

            var prev = arr[k];
            var next = arr[k + 1];
            long drow = next.Start - prev.Start;
            if (drow <= 0) return prev.T;
            double slope = (next.T - prev.T) / drow;
            return prev.T + (row - prev.Start) * slope;
        }

        // 公開情報
        public long HeadTotal   => Interlocked.Read(ref _headTotal);
        public long StoredLines => Volatile.Read(ref _storedLines);

        // 取得API（時刻＝窓先頭行の時刻）
        public bool TryGetLatestWindowPtr(int winW, int winH, int x0,
                                          out IntPtr ptr, out int strideBytes, out double timeSecAtTop)
        {
            if (Volatile.Read(ref _disposed)) throw new ObjectDisposedException(nameof(LineStore));
            ptr = IntPtr.Zero; strideBytes = RowBytes; timeSecAtTop = double.NaN;

            if (winW <= 0 || winH <= 0 || winW > Width) return false;

            long avail = Volatile.Read(ref _storedLines);
            if (avail < winH) return false;

            long topRow = avail - winH;
            return TryGetWindowPtr(topRow, winW, winH, x0, out ptr, out strideBytes, out timeSecAtTop);
        }

        public bool TryGetWindowPtr(long startRow, int winW, int winH, int x0,
                                    out IntPtr ptr, out int strideBytes, out double timeSecAtTop)
        {
            if (Volatile.Read(ref _disposed)) throw new ObjectDisposedException(nameof(LineStore));
            ptr = IntPtr.Zero; strideBytes = RowBytes; timeSecAtTop = double.NaN;

            if (startRow < 0 || winW <= 0 || winH <= 0 || winW > Width) return false;

            long avail = Volatile.Read(ref _storedLines);
            if (startRow + winH > avail) return false;

            int x0c = Clamp(x0, 0, Math.Max(0, Width - winW));
            long byteOffset = startRow * (long)RowBytes + (long)x0c * ElemSizeBytes;
            ptr = (IntPtr)((byte*)_buf + byteOffset);

            timeSecAtTop = RowTimeSec(startRow);
            return true;
        }

        // 互換（時刻なし）
        public bool TryGetLatestWindowPtr(int winW, int winH, int x0, out IntPtr ptr, out int strideBytes)
            => TryGetLatestWindowPtr(winW, winH, x0, out ptr, out strideBytes, out _);
        public bool TryGetWindowPtr(long startRow, int winW, int winH, int x0, out IntPtr ptr, out int strideBytes)
            => TryGetWindowPtr(startRow, winW, winH, x0, out ptr, out strideBytes, out _);

        // Utils
        private static int Clamp(int v, int min, int max)
        {
            if (v < min) return min;
            if (v > max) return max;
            return v;
        }
        private static double DateTimeToUnixSec(DateTime utc)
        {
            var epoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
            return (utc - epoch).TotalSeconds;
        }
    }
}