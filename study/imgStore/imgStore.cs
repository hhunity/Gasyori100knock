using System;
using System.Runtime.InteropServices;
using System.Threading;

namespace YourApp.Imaging
{
    public enum PixelType { U8 = 0, U16 = 1 }

    /// <summary>
    /// ラインセンサー用ラインバッファ（ROI対応版）
    /// - 受信画像の一部(ROI)だけを保持：ソース幅 srcWidth のうち [roiX, roiX+roiW) を保存
    /// - ウォームアップ: 最新 WarmupMax 行を常に 0..WarmupMax-1 に連続配置
    /// - Commit 後: WarmupMax の続きから線形追記（最大 CapacityLines）
    /// - 1 writer (PushBlock) + N readers (TryGet...) 前提（ロックなし公開）
    /// - PushBlock に渡した取得時刻(UTC秒, double)をブロック先頭行に記録
    /// - TryGetWindowPtr で返す時刻は「窓の先頭行の時刻（ブロック境界で線形補間/外挿）」
    /// </summary>
    public unsafe sealed class LineStore : IDisposable
    {
        // ---- 構成（不変） ----
        public int      SourceWidth   { get; }      // 受信元の横幅（例: 2048）
        public int      RoiX          { get; }      // 取り込む開始 x（0..SourceWidth-1）
        public int      Width         { get; }      // 取り込む幅（= ROI 幅）
        public long     CapacityLines { get; }      // 行バッファ容量
        public int      WarmupMax     { get; }      // ウォームアップ行数（コンストラクタで指定）
        public PixelType PixelType    { get; }
        public int      ElemSizeBytes { get; }
        public int      RowBytes      => Width * ElemSizeBytes;
        public int      SourceRowBytes => SourceWidth * ElemSizeBytes;

        // ---- 状態 ----
        private volatile bool _committed;   // false: ウォームアップ, true: 線形追記
        private int _warmupCount;           // 0..WarmupMax（ウォームアップ中のみ）

        private IntPtr _buf;                // [CapacityLines x RowBytes]
        private long   _writeIndex;         // Commit 後の次書込位置（WarmupMax から増える）
        private long   _headTotal;          // 受信成功総行数（統計）
        private long   _storedLines;        // いま連続で使える行数（0..WarmupMax → WarmupMax..CapacityLines）
        private volatile bool _disposed;

        // ---- ブロック時刻（補間用）----
        private struct TimeSeg { public long Start; public double T; } // Start: 0-based 行index
        private TimeSeg[] _segs = new TimeSeg[64];
        private int _segCount = 0;                   // Volatile.Write で公開
        private double _warmupLastTimeSec = double.NaN;

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

            SourceWidth   = srcWidth;
            RoiX          = roiX;
            Width         = roiW;           // 以降は ROI 幅が "Width"
            CapacityLines = capacityLines;
            WarmupMax     = warmupMax;

            PixelType     = pt;
            ElemSizeBytes = (pt == PixelType.U8) ? 1 : 2;

            checked
            {
                long totalBytes = CapacityLines * (long)RowBytes;
                _buf = Marshal.AllocHGlobal((nint)totalBytes);
            }

            _committed   = false;
            _warmupCount = 0;
            _writeIndex  = 0;
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

            AddSeg(0, _warmupLastTimeSec);

            _writeIndex = _warmupCount; // 通常 = WarmupMax
            Volatile.Write(ref _storedLines, _warmupCount);
            Volatile.Write(ref _committed, true);
        }

        // ---- PushBlock（取得時刻つき）----
        // 互換：時刻未指定→現在UTC
        public bool PushBlock(IntPtr src, int rows, int srcStrideBytes)
            => PushBlock(src, rows, srcStrideBytes, DateTimeToUnixSec(DateTime.UtcNow));

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

        // ---- ウォームアップ（最新 WarmupMax 行のみ保持）----
        private void PushWarmup(IntPtr src, int rows, int srcStrideBytes, double timeSec)
        {
            byte* sBase = (byte*)src;
            byte* dBase = (byte*)_buf;
            int   roiByteOffset = RoiX * ElemSizeBytes;

            int filled = _warmupCount; // 0..WarmupMax

            // 1) 満杯まで埋める
            if (filled < WarmupMax)
            {
                int need = WarmupMax - filled;
                int take = rows < need ? rows : need;

                // ROIコピー（1行ずつ。roiX>0 なら必ず行ループ）
                for (int i = 0; i < take; i++)
                {
                    byte* srcLine = sBase + (long)i * srcStrideBytes + roiByteOffset;
                    byte* dstLine = dBase + (long)(filled + i) * RowBytes;
                    Buffer.MemoryCopy(srcLine, dstLine, RowBytes, RowBytes);
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

            // 2) 満杯（WarmupMax 行）状態の更新
            if (rows >= WarmupMax)
            {
                // ブロック末尾 WarmupMax 行で置換
                byte* tail = sBase + (long)(rows - WarmupMax) * srcStrideBytes;
                for (int i = 0; i < WarmupMax; i++)
                {
                    byte* srcLine = tail + (long)i * srcStrideBytes + roiByteOffset;
                    byte* dstLine = dBase + (long)i * RowBytes;
                    Buffer.MemoryCopy(srcLine, dstLine, RowBytes, RowBytes);
                }
                Volatile.Write(ref _storedLines, WarmupMax);
            }
            else // 0 < rows < WarmupMax
            {
                int shift = rows;                 // 左詰め量
                int keep  = WarmupMax - shift;    // 残す本数

                // 前詰め（オーバーラップあり：y昇順）
                for (int y = 0; y < keep; y++)
                {
                    byte* srcRow = dBase + (long)(y + shift) * RowBytes;
                    byte* dstRow = dBase + (long)y * RowBytes;
                    Buffer.MemoryCopy(srcRow, dstRow, RowBytes, RowBytes);
                }

                // 末尾に rows 行の ROI を配置
                for (int i = 0; i < rows; i++)
                {
                    byte* srcLine = sBase + (long)i * srcStrideBytes + roiByteOffset;
                    byte* dstLine = dBase + (long)(keep + i) * RowBytes;
                    Buffer.MemoryCopy(srcLine, dstLine, RowBytes, RowBytes);
                }

                Volatile.Write(ref _storedLines, WarmupMax);
            }

            _warmupLastTimeSec = timeSec; // 最新ブロックの時刻
        }

        // ---- Commit 後：線形追記。先頭行の時刻をセグメントに記録 ----
        private bool PushLinear(IntPtr src, int rows, int srcStrideBytes, double timeSec)
        {
            long remain = CapacityLines - _writeIndex;
            if (remain <= 0) return false;

            int can = (int)Math.Min(remain, rows);

            // このブロックの先頭行と時刻を追加（補間用）
            long startOfBlock = _writeIndex;
            AddSeg(startOfBlock, timeSec);

            byte* sBase = (byte*)src;
            byte* dBase = (byte*)_buf + _writeIndex * RowBytes;
            int   roiByteOffset = RoiX * ElemSizeBytes;

            // ROIコピー（通常は1行ずつ。roiX==0 かつ srcStrideBytes==RowBytes の場合のみ一括最適化可）
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

            return can == rows; // 途中満了なら false
        }

        // ---- セグメント管理 ----
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
            _segs[n].T = t;
            Volatile.Write(ref _segCount, n + 1);
        }

        // ---- 行 index の時刻をセグメントで線形補間（なければ外挿/定数）----
        private double RowTimeSec(long row)
        {
            int n = Volatile.Read(ref _segCount);
            if (n == 0)
            {
                return double.IsNaN(_warmupLastTimeSec) ? DateTimeToUnixSec(DateTime.UtcNow)
                                                        : _warmupLastTimeSec;
            }

            var arr = _segs;

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
                    return p2.T + (row - p2.Start) * a;
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

        // ---- 公開情報 ----
        public long HeadTotal   => Interlocked.Read(ref _headTotal);
        public long StoredLines => Volatile.Read(ref _storedLines);

        // ---- 取得API（窓の先頭行の時刻も返す）----
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

        // 互換（時刻を返さない版）
        public bool TryGetLatestWindowPtr(int winW, int winH, int x0, out IntPtr ptr, out int strideBytes)
            => TryGetLatestWindowPtr(winW, winH, x0, out ptr, out strideBytes, out _);

        public bool TryGetWindowPtr(long startRow, int winW, int winH, int x0,
                                    out IntPtr ptr, out int strideBytes)
            => TryGetWindowPtr(startRow, winW, winH, x0, out ptr, out strideBytes, out _);

        // ---- ユーティリティ ----
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