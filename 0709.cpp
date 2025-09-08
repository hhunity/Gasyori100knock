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