
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