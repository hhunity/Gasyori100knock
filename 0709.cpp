
// ===== 単一PNG保存API（分割なし） =====

// 最新 rows 行を 1枚のPNGに保存（分割なし）
public unsafe bool SaveLatestToPngSingle(int rows, int x0, int winW, string path,
                                         out string error, bool normalize16To8 = true)
{
    error = null;
    if (rows <= 0) { error = "rows <= 0"; return false; }
    long avail = StoredLines;
    if (avail < rows) { error = "not enough lines"; return false; }
    long startRow = avail - rows;
    return SaveRangeToPngSingle(startRow, rows, x0, winW, path, out error, normalize16To8);
}

// 任意範囲を 1枚のPNGに保存（分割なし）
public unsafe bool SaveRangeToPngSingle(long startRow, int rows, int x0, int winW, string path,
                                        out string error, bool normalize16To8 = true)
{
    error = null;
    if (_disposed) { error = "disposed"; return false; }
    if (startRow < 0 || rows <= 0) { error = "invalid start/rows"; return false; }
    if (winW <= 0 || winW > Width) { error = "invalid width"; return false; }

    long avail = StoredLines;
    if (startRow + rows > avail) { error = "range exceeds stored lines"; return false; }

    IntPtr p; int stride;
    if (!TryGetWindowPtr(startRow, winW, rows, x0, out p, out stride))
    {
        error = "TryGetWindowPtr failed"; return false;
    }

    string? dir = Path.GetDirectoryName(path);
    if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

    bool ok;
    if (PixelType == PixelType.U8)
        ok = Save8bppGraySingle((byte*)p, rows, winW, stride, path, out error);
    else
        ok = Save16to8Single((byte*)p, rows, winW, stride, path, out error, normalize16To8);

    return ok;
}

// 8bitグレースケールを 1枚のPNGへ（分割なし）
private static unsafe bool Save8bppGraySingle(byte* srcTop, int rows, int cols, int srcStrideBytes,
                                              string path, out string? error)
{
    error = null;
    try
    {
        using (var bmp = new Bitmap(cols, rows, PixelFormat.Format8bppIndexed))
        {
            // グレーパレット
            var pal = bmp.Palette;
            for (int i = 0; i < 256; i++) pal.Entries[i] = Color.FromArgb(i, i, i);
            bmp.Palette = pal;

            var rect = new Rectangle(0, 0, cols, rows);
            BitmapData bd = bmp.LockBits(rect, ImageLockMode.WriteOnly, bmp.PixelFormat);
            try
            {
                byte* dstBase = (byte*)bd.Scan0;
                int dstStride = bd.Stride; // 4バイト境界
                for (int y = 0; y < rows; y++)
                {
                    Buffer.MemoryCopy(
                        srcTop + (long)y * srcStrideBytes,
                        dstBase + (long)y * dstStride,
                        dstStride,
                        cols
                    );
                }
            }
            finally { bmp.UnlockBits(bd); }

            // temp に書いてから置換（汎用エラー対策）
            string tmp = path + ".tmp_" + Guid.NewGuid().ToString("N");
            using (var fs = new FileStream(tmp, FileMode.CreateNew, FileAccess.Write, FileShare.None, 64 * 1024, FileOptions.WriteThrough))
            {
                bmp.Save(fs, ImageFormat.Png);
                fs.Flush(true);
            }
            TryReplace(path, tmp);
        }
        return true;
    }
    catch (Exception ex) { error = ex.ToString(); return false; }
}

// 16bit → 8bitに変換して 1枚のPNGへ（分割なし）
// normalize16To8=false: 上位8bit抽出 / true: min-max正規化
private static unsafe bool Save16to8Single(byte* srcTop, int rows, int cols, int srcStrideBytes,
                                           string path, out string? error, bool normalize16To8)
{
    error = null;
    try
    {
        using (var bmp = new Bitmap(cols, rows, PixelFormat.Format8bppIndexed))
        {
            var pal = bmp.Palette;
            for (int i = 0; i < 256; i++) pal.Entries[i] = Color.FromArgb(i, i, i);
            bmp.Palette = pal;

            var rect = new Rectangle(0, 0, cols, rows);
            BitmapData bd = bmp.LockBits(rect, ImageLockMode.WriteOnly, bmp.PixelFormat);
            try
            {
                byte* dstBase = (byte*)bd.Scan0;
                int dstStride = bd.Stride;

                if (!normalize16To8)
                {
                    // 上位8bit抽出（Little Endian）
                    for (int y = 0; y < rows; y++)
                    {
                        byte* s = srcTop + (long)y * srcStrideBytes;
                        byte* d = dstBase + (long)y * dstStride;
                        for (int x = 0; x < cols; x++)
                            d[x] = s[x * 2 + 1];
                    }
                }
                else
                {
                    // min-max 正規化
                    ushort minV = ushort.MaxValue, maxV = 0;
                    for (int y = 0; y < rows; y++)
                    {
                        byte* s = srcTop + (long)y * srcStrideBytes;
                        for (int x = 0; x < cols; x++)
                        {
                            ushort v = (ushort)(s[x * 2] | (s[x * 2 + 1] << 8));
                            if (v < minV) minV = v;
                            if (v > maxV) maxV = v;
                        }
                    }
                    double scale = (maxV > minV) ? 255.0 / (maxV - minV) : 1.0;

                    for (int y = 0; y < rows; y++)
                    {
                        byte* s = srcTop + (long)y * srcStrideBytes;
                        byte* d = dstBase + (long)y * dstStride;
                        for (int x = 0; x < cols; x++)
                        {
                            ushort v = (ushort)(s[x * 2] | (s[x * 2 + 1] << 8));
                            int u8 = (int)((v - minV) * scale + 0.5);
                            if (u8 < 0) u8 = 0; else if (u8 > 255) u8 = 255;
                            d[x] = (byte)u8;
                        }
                    }
                }
            }
            finally { bmp.UnlockBits(bd); }

            string tmp = path + ".tmp_" + Guid.NewGuid().ToString("N");
            using (var fs = new FileStream(tmp, FileMode.CreateNew, FileAccess.Write, FileShare.None, 64 * 1024, FileOptions.WriteThrough))
            {
                bmp.Save(fs, ImageFormat.Png);
                fs.Flush(true);
            }
            TryReplace(path, tmp);
        }
        return true;
    }
    catch (Exception ex) { error = ex.ToString(); return false; }
}

private static void TryReplace(string dest, string tmp)
{
    try { if (File.Exists(dest)) File.Delete(dest); } catch { /* ファイルロック中などは上書き不可 */ }
    File.Move(tmp, dest);
}
// ===== ここまで：単一PNG保存API =====

// ===== ここから: PNG 保存API =====

// 最新 rows 行を PNG 保存（必要なら複数ファイルに分割）
// 例: SaveLatestToPng(725, 0, 725, @"C:\out\latest.png");
public unsafe bool SaveLatestToPng(int rows, int x0, int winW, string basePath,
                                   int maxRowsPerImage /*例: 16384*/, out string error,
                                   bool normalize16To8 = true)
{
    error = null;
    if (rows <= 0) { error = "rows <= 0"; return false; }

    long avail = StoredLines;
    if (avail < rows) { error = "not enough lines"; return false; }

    long startRow = avail - rows;
    return SaveRangeToPng(startRow, rows, x0, winW, basePath, maxRowsPerImage, out error, normalize16To8);
}

// 任意範囲を PNG 保存（必要なら複数ファイルに分割）
// basePath が "foo.png" のとき、分割時は "foo_p0001.png" のように出力
public unsafe bool SaveRangeToPng(long startRow, int rows, int x0, int winW, string basePath,
                                  int maxRowsPerImage /*例: 16384*/, out string error,
                                  bool normalize16To8 = true)
{
    error = null;
    if (_disposed) { error = "disposed"; return false; }
    if (startRow < 0 || rows <= 0) { error = "invalid start/rows"; return false; }
    if (winW <= 0 || winW > Width) { error = "invalid width"; return false; }

    long avail = StoredLines;
    if (startRow + rows > avail) { error = "range exceeds stored lines"; return false; }

    IntPtr p; int stride;
    if (!TryGetWindowPtr(startRow, winW, rows, x0, out p, out stride))
    {
        error = "TryGetWindowPtr failed"; return false;
    }

    // 出力先フォルダ作成
    string dir = Path.GetDirectoryName(basePath);
    if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

    if (PixelType == PixelType.U8)
    {
        Save8bppGrayStripes((byte*)p, rows, winW, stride, basePath, maxRowsPerImage);
        return true;
    }
    else // U16
    {
        if (normalize16To8)
        {
            Save16to8Stripes((byte*)p, rows, winW, stride, basePath, maxRowsPerImage, false /*minmax*/);
            return true;
        }
        else
        {
            // 上位8bit抽出ではなく min-max 正規化したい場合は第6引数を true に
            Save16to8Stripes((byte*)p, rows, winW, stride, basePath, maxRowsPerImage, true /*minmax*/);
            return true;
        }
    }
}

// ===== 内部実装: 8bitグレースケール（ストライプ分割） =====
private static unsafe void Save8bppGrayStripes(byte* srcTop, int rows, int cols, int srcStrideBytes,
                                               string basePath, int maxRowsPerImage)
{
    int parts = (rows + maxRowsPerImage - 1) / maxRowsPerImage;
    string dir = Path.GetDirectoryName(basePath);
    string name = Path.GetFileNameWithoutExtension(basePath);
    string ext = Path.GetExtension(basePath);

    for (int pi = 0; pi < parts; pi++)
    {
        int rowStart = pi * maxRowsPerImage;
        int h = Math.Min(maxRowsPerImage, rows - rowStart);
        string outPath = (parts == 1) ? basePath :
            Path.Combine(dir ?? "", name + "_p" + (pi + 1).ToString("0000") + ext);

        using (var bmp = new Bitmap(cols, h, PixelFormat.Format8bppIndexed))
        {
            // 0..255 のグレーパレット
            var pal = bmp.Palette;
            for (int i = 0; i < 256; i++) pal.Entries[i] = Color.FromArgb(i, i, i);
            bmp.Palette = pal;

            var rect = new Rectangle(0, 0, cols, h);
            BitmapData bd = bmp.LockBits(rect, ImageLockMode.WriteOnly, bmp.PixelFormat);
            try
            {
                byte* dstBase = (byte*)bd.Scan0;
                int dstStride = bd.Stride; // 4バイト境界

                byte* srcStripe = srcTop + (long)rowStart * srcStrideBytes;
                for (int y = 0; y < h; y++)
                {
                    Buffer.MemoryCopy(
                        srcStripe + (long)y * srcStrideBytes,
                        dstBase   + (long)y * dstStride,
                        dstStride,
                        cols
                    );
                }
            }
            finally { bmp.UnlockBits(bd); }

            // temp に書いて Move（GDI+汎用エラー回避策）
            string tmp = outPath + ".tmp_" + Guid.NewGuid().ToString("N");
            using (var fs = new FileStream(tmp, FileMode.CreateNew, FileAccess.Write, FileShare.None, 64 * 1024, FileOptions.WriteThrough))
            {
                bmp.Save(fs, ImageFormat.Png);
                fs.Flush(true);
            }
            TryReplace(outPath, tmp);
        }
    }
}

// ===== 内部実装: 16bit→8bit 変換してストライプ保存 =====
private static unsafe void Save16to8Stripes(byte* srcTop, int rows, int cols, int srcStrideBytes,
                                            string basePath, int maxRowsPerImage, bool useMinMaxScale)
{
    int parts = (rows + maxRowsPerImage - 1) / maxRowsPerImage;
    string dir = Path.GetDirectoryName(basePath);
    string name = Path.GetFileNameWithoutExtension(basePath);
    string ext = Path.GetExtension(basePath);

    for (int pi = 0; pi < parts; pi++)
    {
        int rowStart = pi * maxRowsPerImage;
        int h = Math.Min(maxRowsPerImage, rows - rowStart);
        string outPath = (parts == 1) ? basePath :
            Path.Combine(dir ?? "", name + "_p" + (pi + 1).ToString("0000") + ext);

        using (var bmp = new Bitmap(cols, h, PixelFormat.Format8bppIndexed))
        {
            var pal = bmp.Palette;
            for (int i = 0; i < 256; i++) pal.Entries[i] = Color.FromArgb(i, i, i);
            bmp.Palette = pal;

            var rect = new Rectangle(0, 0, cols, h);
            BitmapData bd = bmp.LockBits(rect, ImageLockMode.WriteOnly, bmp.PixelFormat);
            try
            {
                byte* dstBase = (byte*)bd.Scan0;
                int dstStride = bd.Stride;

                byte* srcStripe = srcTop + (long)rowStart * srcStrideBytes;

                if (!useMinMaxScale)
                {
                    // 単純に上位8bit抽出（Little Endian想定）
                    for (int y = 0; y < h; y++)
                    {
                        byte* s = srcStripe + (long)y * srcStrideBytes;
                        byte* d = dstBase   + (long)y * dstStride;
                        for (int x = 0; x < cols; x++)
                        {
                            d[x] = s[x * 2 + 1]; // HIバイト
                        }
                    }
                }
                else
                {
                    // min-max 正規化
                    ushort minV = ushort.MaxValue, maxV = 0;
                    for (int y = 0; y < h; y++)
                    {
                        byte* s = srcStripe + (long)y * srcStrideBytes;
                        for (int x = 0; x < cols; x++)
                        {
                            ushort v = (ushort)(s[x * 2] | (s[x * 2 + 1] << 8));
                            if (v < minV) minV = v;
                            if (v > maxV) maxV = v;
                        }
                    }
                    double scale = (maxV > minV) ? 255.0 / (maxV - minV) : 1.0;

                    for (int y = 0; y < h; y++)
                    {
                        byte* s = srcStripe + (long)y * srcStrideBytes;
                        byte* d = dstBase   + (long)y * dstStride;
                        for (int x = 0; x < cols; x++)
                        {
                            ushort v = (ushort)(s[x * 2] | (s[x * 2 + 1] << 8));
                            int u8 = (int)((v - minV) * scale + 0.5);
                            if (u8 < 0) u8 = 0; else if (u8 > 255) u8 = 255;
                            d[x] = (byte)u8;
                        }
                    }
                }
            }
            finally { bmp.UnlockBits(bd); }

            string tmp = outPath + ".tmp_" + Guid.NewGuid().ToString("N");
            using (var fs = new FileStream(tmp, FileMode.CreateNew, FileAccess.Write, FileShare.None, 64 * 1024, FileOptions.WriteThrough))
            {
                bmp.Save(fs, ImageFormat.Png);
                fs.Flush(true);
            }
            TryReplace(outPath, tmp);
        }
    }
}

private static void TryReplace(string dest, string tmp)
{
    try { if (File.Exists(dest)) File.Delete(dest); } catch { /* ignore */ }
    File.Move(tmp, dest);
}

// ===== ここまで: PNG 保存API =====


// LineStore_Warmup6.cs
// 目的：PushBlock で受け取り、
//  (A) ウォームアップ中は最大6行だけ保持（超えた分は毎回スクロールして 0..5 に最新6行を連続配置）
//  (B) Commit() 呼び出しで、6行の続きから線形に CapacityLines まで追記（以降は通常の直線バッファ）
// 取得：最新から/任意行から、ゼロコピーで先頭ポインタ＋ストライドを返す
using System;
using System.Runtime.InteropServices;
using System.Threading;

namespace YourApp.Imaging
{
    public enum PixelType { U8 = 0, U16 = 1 }

    public unsafe sealed class LineStore : IDisposable
    {
        public int  Width          { get; }
        public long CapacityLines  { get; }
        public PixelType PixelType { get; }
        public int  ElemSizeBytes  { get; }
        public int  RowBytes       => Width * ElemSizeBytes;

        // モード制御
        private const int WARMUP_MAX = 6;
        private volatile bool _committed;     // false: ウォームアップ(最大6行) / true: 線形追記
        private int _warmupCount;             // 現在ウォームアップ領域にある行数 (0..6)

        // バッファ
        private IntPtr _buf;                  // [CapacityLines x Width] 連続領域
        private long _headTotal;              // 受信総行数（単調増加、統計用）
        private long _writeIndex;             // 線形追記時の次書込行（commit後は 6 から増える）
        private long _storedLines;            // 現在バッファ内に有効に並んでいる行数（commit前は 0..6、commit後は 6..CapacityLines）
        private bool _disposed;

        public LineStore(int width, long capacityLines, PixelType pt)
        {
            if (IntPtr.Size == 4) throw new PlatformNotSupportedException("x64 で使用してください。");
            if (width <= 0 || capacityLines < WARMUP_MAX) throw new ArgumentOutOfRangeException();

            Width = width;
            CapacityLines = capacityLines;
            PixelType = pt;
            ElemSizeBytes = (pt == PixelType.U8) ? 1 : 2;

            checked
            {
                long totalBytes = CapacityLines * (long)RowBytes;
                _buf = Marshal.AllocHGlobal((nint)totalBytes);
            }

            _committed   = false;
            _warmupCount = 0;
            _writeIndex  = 0;          // commit 前は未使用、commit 時に 6 に設定
            _storedLines = 0;
            _headTotal   = 0;
        }

        public void Dispose()
        {
            if (_disposed) return;
            Marshal.FreeHGlobal(_buf);
            _buf = IntPtr.Zero;
            _disposed = true;
        }

        /// <summary>
        /// ウォームアップを終了し、現在の6行の「続き」から線形追記モードに移行します。
        /// この呼び出し以降、PushBlock は 6行目以降 (= index 6 から) へ連続書き込みします。
        /// </summary>
        public void Commit()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(LineStore));
            if (_committed) return;

            // ウォームアップ中は最新行群が 0.._warmupCount-1 に時系列順で連続配置済み（常にスクロールしているため）
            // よってそのまま index=_warmupCount の位置から線形追記を開始できる
            _writeIndex = _warmupCount;             // 通常 6（まだ6未満ならその数）
            _storedLines = _warmupCount;            // 有効行数を引き継ぎ
            _committed = true;                      // モード切替
        }

        /// <summary>
        /// 2048xH ブロックを取り込み。ウォームアップ中は最大6行に抑えつつスクロール、Commit 後は線形追記。
        /// </summary>
        public bool PushBlock(IntPtr src, int rows, int strideBytes)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(LineStore));
            if (src == IntPtr.Zero) throw new ArgumentNullException(nameof(src));
            if (rows <= 0) return true;
            if (strideBytes < RowBytes) throw new ArgumentException("stride too small");

            if (!_committed)
            {
                PushWarmup(src, rows, strideBytes);
                Interlocked.Add(ref _headTotal, rows);
                // _storedLines は PushWarmup 内で更新
                return true;
            }
            else
            {
                return PushLinear(src, rows, strideBytes);
            }
        }

        // --- ウォームアップ：最大6行。超過分は 0..5 を毎回左詰めスクロールして末尾に新規行を置く ---
        private void PushWarmup(IntPtr src, int rows, int strideBytes)
        {
            byte* sBase = (byte*)src;
            byte* dBase = (byte*)_buf; // 先頭から WARMUP_MAX 行分だけ使用

            for (int i = 0; i < rows; i++)
            {
                if (_warmupCount < WARMUP_MAX)
                {
                    // まだ6行未満：末尾に追加
                    byte* dst = dBase + (long)_warmupCount * RowBytes;
                    byte* srcLine = sBase + (long)i * strideBytes;
                    CopyRow(srcLine, dst);
                    _warmupCount++;
                    _storedLines = _warmupCount;
                }
                else
                {
                    // すでに6行ある：0..4 を左詰め（行ごとコピー）、5 に新規行
                    // 注意：オーバーラップコピーなので 0→1 の順で1行ずつコピーすれば安全
                    for (int y = 0; y < WARMUP_MAX - 1; y++)
                    {
                        byte* srcRow = dBase + (long)(y + 1) * RowBytes;
                        byte* dstRow = dBase + (long)(y)     * RowBytes;
                        CopyRow(srcRow, dstRow);
                    }
                    byte* srcLine = sBase + (long)i * strideBytes;
                    byte* dstLast = dBase + (long)(WARMUP_MAX - 1) * RowBytes;
                    CopyRow(srcLine, dstLast);
                    // _warmupCount stays 6, _storedLines stays 6
                }
            }
        }

        // --- コミット後：線形追記（index=6 から CapacityLines-1 まで） ---
        private bool PushLinear(IntPtr src, int rows, int strideBytes)
        {
            long remain = CapacityLines - _writeIndex;
            if (remain <= 0) return false; // これ以上は詰めない（満了）

            int can = (int)Math.Min(remain, rows);
            byte* sBase = (byte*)src;
            byte* dBase = (byte*)_buf + _writeIndex * RowBytes;

            if (strideBytes == RowBytes)
            {
                long bytes = (long)can * RowBytes;
                Buffer.MemoryCopy(sBase, dBase, bytes, bytes);
            }
            else
            {
                for (int i = 0; i < can; i++)
                {
                    byte* srcLine = sBase + (long)i * strideBytes;
                    byte* dstLine = dBase + (long)i * RowBytes;
                    CopyRow(srcLine, dstLine);
                }
            }

            Interlocked.Add(ref _headTotal, can);
            _writeIndex += can;
            long newStored = _writeIndex;                 // 先頭から _writeIndex 行が有効（0..5 は初期の6行）
            if (newStored > _storedLines) _storedLines = newStored;

            return can == rows; // 余りがあるなら false（満了）
        }

        private void CopyRow(byte* src, byte* dst)
        {
            Buffer.MemoryCopy(src, dst, RowBytes, RowBytes);
        }

        // === 情報系 ===
        public long HeadTotal   => Interlocked.Read(ref _headTotal); // 受信総行数（統計）
        public long StoredLines => Interlocked.Read(ref _storedLines); // 現在バッファ内の有効行数

        /// <summary>
        /// 最新 winH 行の x0..x0+winW-1 をゼロコピーで指す（ウォームアップ中は最大6まで）。
        /// </summary>
        public bool TryGetLatestWindowPtr(int winW, int winH, int x0, out IntPtr ptr, out int strideBytes)
        {
            ptr = IntPtr.Zero; strideBytes = RowBytes;
            if (_disposed) throw new ObjectDisposedException(nameof(LineStore));
            if (winW <= 0 || winH <= 0 || winW > Width) return false;

            long avail = StoredLines;
            if (avail < winH) return false;

            long topRow = avail - winH; // 0-based
            return TryGetWindowPtr(topRow, winW, winH, x0, out ptr, out strideBytes);
        }

        /// <summary>
        /// 任意の開始行 startRow（0基点）から winH 行、横 x0.. の窓をゼロコピーで返す。
        /// </summary>
        public bool TryGetWindowPtr(long startRow, int winW, int winH, int x0,
                                    out IntPtr ptr, out int strideBytes)
        {
            ptr = IntPtr.Zero; strideBytes = RowBytes;
            if (_disposed) throw new ObjectDisposedException(nameof(LineStore));
            if (startRow < 0 || winW <= 0 || winH <= 0 || winW > Width) return false;

            long avail = StoredLines;
            if (startRow + winH > avail) return false;

            int x0c = Clamp(x0, 0, Math.Max(0, Width - winW));
            long byteOffset = startRow * (long)RowBytes + (long)x0c * ElemSizeBytes;
            ptr = (IntPtr)((byte*)_buf + byteOffset);
            return true;
        }

        private static int Clamp(int v, int min, int max)
        {
            if (v < min) return min;
            if (v > max) return max;
            return v;
        }
    }
}



// ===== PNG保存：範囲指定（full幅or任意幅） =====
        // 使い方例：SaveRangeToPng(0, (int)HeadLines, 0, Width, "all.png")
        public bool SaveRangeToPng(long startRow, int rows, int x0, int winW, string path,
                                   bool normalize16To8 = true)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(LineStore));
            if (string.IsNullOrWhiteSpace(path)) throw new ArgumentNullException(nameof(path));
            if (startRow < 0 || rows <= 0 || winW <= 0 || winW > Width) return false;

            long h = Interlocked.Read(ref _head);
            if (h < startRow + rows) return false; // データ不足

            int  x0c = Clamp(x0, 0, Math.Max(0, Width - winW));
            byte* srcTop = (byte*)_buf + startRow * RowBytes + (long)x0c * ElemSizeBytes;

            switch (PixelType)
            {
                case PixelType.U8:
                    return Save8bppGrayscale(srcTop, rows, winW, RowBytes, path);

                case PixelType.U16:
                    if (normalize16To8)
                        return Save16to8(srcTop, rows, winW, RowBytes, path);
                    else
                        throw new NotSupportedException("16bitそのままPNG保存は標準APIでは扱いにくいです（ImageSharp等をご利用ください）。");

                default:
                    throw new NotSupportedException("Unknown pixel type");
            }
        }

        // 最新 rows 行をPNG保存（幅は winW、左端 x0）
        public bool SaveLatestToPng(int rows, int x0, int winW, string path,
                                    bool normalize16To8 = true)
        {
            if (rows <= 0) return false;
            long h = Interlocked.Read(ref _head);
            if (h < rows) return false;
            long startRow = h - rows;
            return SaveRangeToPng(startRow, rows, x0, winW, path, normalize16To8);
        }

        // ===== 内部: 8bit グレースケール PNG =====
        private static bool Save8bppGrayscale(byte* srcTop, int rows, int cols, int srcStrideBytes, string path)
        {
            using var bmp = new Bitmap(cols, rows, PixelFormat.Format8bppIndexed);
            // グレーパレット設定
            ColorPalette pal = bmp.Palette;
            for (int i = 0; i < 256; i++) pal.Entries[i] = Color.FromArgb(i, i, i);
            bmp.Palette = pal;

            var rect = new Rectangle(0, 0, cols, rows);
            BitmapData bd = bmp.LockBits(rect, ImageLockMode.WriteOnly, bmp.PixelFormat);
            try
            {
                byte* dstRow = (byte*)bd.Scan0;
                int   dstStride = bd.Stride;
                for (int y = 0; y < rows; y++)
                {
                    Buffer.MemoryCopy(srcTop + (long)y * srcStrideBytes,
                                      dstRow + (long)y * dstStride,
                                      dstStride, cols);
                }
            }
            finally { bmp.UnlockBits(bd); }

            bmp.Save(path, System.Drawing.Imaging.ImageFormat.Png);
            return true;
        }

        // ===== 内部: 16bit → 8b

public bool TryGetWindowPtr(long startRow, int winW, int winH, int x0,
                            out IntPtr ptr, out int strideBytes)
{
    if (_disposed) throw new ObjectDisposedException(nameof(LineStore));
    ptr = IntPtr.Zero;
    strideBytes = RowBytes;

    if (winW <= 0 || winH <= 0 || winW > Width) return false;
    if (startRow < 0) return false;

    long h = Interlocked.Read(ref _head);
    // まだ startRow+winH 行まで入っていない場合は失敗
    if (h < startRow + winH) return false;

    int x0Clamped = (x0 < 0) ? 0 : (x0 > Width - winW ? Width - winW : x0);
    long byteOffset = startRow * RowBytes + (long)x0Clamped * ElemSizeBytes;
    ptr = (IntPtr)((byte*)_buf + byteOffset);
    return true;
}



public static int Clamp(int value, int min, int max)
{
    if (value < min) return min;
    if (value > max) return max;
    return value;
}


// LineStore.cs (エラーハンドリング強化版)
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;

namespace YourApp.Imaging
{
    public enum PixelType { U8 = 0, U16 = 1 }

    public unsafe sealed class LineStore : IDisposable
    {
        public int Width { get; }
        public long CapacityLines { get; }
        public PixelType PixelType { get; }
        public int ElemSizeBytes { get; }
        public int RowBytes => Width * ElemSizeBytes;

        private IntPtr _buf;
        private volatile long _head;
        private bool _disposed;

        /// <summary>
        /// 直接 new する場合は OutOfMemoryException を上位へ投げます。
        /// 失敗時に落ち着いて扱いたい場合は TryCreate を使ってください。
        /// </summary>
        public LineStore(int width, long capacityLines, PixelType pt)
        {
            if (IntPtr.Size == 4) throw new PlatformNotSupportedException("x64 ビルドで使用してください。");
            if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
            if (capacityLines <= 0) throw new ArgumentOutOfRangeException(nameof(capacityLines));

            Width = width;
            CapacityLines = capacityLines;
            PixelType = pt;
            ElemSizeBytes = (pt == PixelType.U8) ? 1 : 2;

            checked
            {
                long totalBytes = CapacityLines * (long)RowBytes;
                if (totalBytes <= 0) throw new ArgumentOutOfRangeException(nameof(capacityLines), "サイズ計算がオーバーフローしました。");

                try
                {
                    _buf = Marshal.AllocHGlobal((nint)totalBytes); // 失敗時は OutOfMemoryException
                }
                catch (OutOfMemoryException oom)
                {
                    throw new InsufficientMemoryException(
                        $"LineStore: {totalBytes / (1024.0 * 1024.0):F1} MiB の確保に失敗しました。width={Width}, lines={CapacityLines}, bytes/row={RowBytes}.",
                        oom);
                }
            }

            // 初期化は任意（巨大確保時は重いので通常は省略推奨）
            // Unsafe.InitBlockUnaligned((void*)_buf, 0, (uint)Math.Min(totalBytes, int.MaxValue));

            _head = 0;
        }

        /// <summary>
        /// 例外を投げない作成API。成功すれば store!=null で true。失敗時は false と error に理由。
        /// </summary>
        public static bool TryCreate(int width, long capacityLines, PixelType pt,
                                     out LineStore? store, out string? error)
        {
            store = null; error = null;
            try
            {
                store = new LineStore(width, capacityLines, pt);
                return true;
            }
            catch (Exception ex) when (ex is OutOfMemoryException || ex is InsufficientMemoryException || ex is ArgumentOutOfRangeException)
            {
                error = ex.Message;
                return false;
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            Marshal.FreeHGlobal(_buf);
            _buf = IntPtr.Zero;
            _disposed = true;
        }

        public bool PushBlock(IntPtr src, int rows, int strideBytes)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(LineStore));
            if (src == IntPtr.Zero) throw new ArgumentNullException(nameof(src));
            if (rows <= 0) return true;
            if (strideBytes < RowBytes) throw new ArgumentException("stride too small", nameof(strideBytes));

            long start = _head;
            long end = start + rows;
            if (end > CapacityLines) return false; // 固定長: これ以上は貯めない

            byte* dstBase = (byte*)_buf + start * RowBytes;
            byte* srcBase = (byte*)src;

            if (strideBytes == RowBytes)
            {
                long bytes = (long)rows * RowBytes;
                Buffer.MemoryCopy(srcBase, dstBase, bytes, bytes);
            }
            else
            {
                for (int i = 0; i < rows; i++)
                {
                    Buffer.MemoryCopy(
                        srcBase + (long)i * strideBytes,
                        dstBase + (long)i * RowBytes,
                        RowBytes, RowBytes);
                }
            }

            Interlocked.Exchange(ref _head, end); // publish
            return true;
        }

        public long HeadLines => Interlocked.Read(ref _head);

        public bool TryGetLatestWindowPtr(int winW, int winH, int x0, out IntPtr ptr, out int strideBytes)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(LineStore));
            ptr = IntPtr.Zero;
            strideBytes = RowBytes;

            if (winW <= 0 || winH <= 0 || winW > Width) return false;

            long h = Interlocked.Read(ref _head);
            if (h < winH) return false;

            int x0Clamped = Math.Clamp(x0, 0, Math.Max(0, Width - winW));
            long topRow = h - winH;

            long byteOffset = topRow * RowBytes + (long)x0Clamped * ElemSizeBytes;
            ptr = (IntPtr)((byte*)_buf + byteOffset);
            return true;
        }
    }
}