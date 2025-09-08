// LineStore.cs
// 固定長の巨大バッファへ 2048xH ブロックを追記し、
// 最新の winW x winH 窓を “ポインタ＋ストライド” でゼロコピー取得するためのユーティリティ。
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;

namespace YourApp.Imaging
{
    public enum PixelType { U8 = 0, U16 = 1 }

    public unsafe sealed class LineStore : IDisposable
    {
        public int Width { get; }                 // 例: 2048
        public long CapacityLines { get; }        // 例: 512行/秒 × 保持秒数
        public PixelType PixelType { get; }
        public int ElemSizeBytes { get; }         // 1(8bit) or 2(16bit)
        public int RowBytes => Width * ElemSizeBytes;

        private IntPtr _buf;                      // [CapacityLines x Width] 連続領域（row-major）
        private volatile long _head;              // これまでに書き込んだ総行数（0..CapacityLines）
        private bool _disposed;

        /// <summary>
        /// 固定長の線形ストアを作成します。先頭から詰めるだけで wrap しません。
        /// </summary>
        /// <param name="width">画素横幅（例: 2048）</param>
        /// <param name="capacityLines">保持する総行数（例: 512 * 保持秒）</param>
        /// <param name="pt">画素型（8bit/16bit）</param>
        public LineStore(int width, long capacityLines, PixelType pt)
        {
            if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
            if (capacityLines <= 0) throw new ArgumentOutOfRangeException(nameof(capacityLines));

            Width = width;
            CapacityLines = capacityLines;
            PixelType = pt;
            ElemSizeBytes = (pt == PixelType.U8) ? 1 : 2;

            long totalBytes = CapacityLines * RowBytes;
            if (IntPtr.Size == 4)
                throw new PlatformNotSupportedException("x64 ビルドで使用してください。");
            if (totalBytes <= 0)
                throw new ArgumentOutOfRangeException(nameof(capacityLines), "容量が大きすぎるかオーバーフローしました。");

            _buf = Marshal.AllocHGlobal(new IntPtr(totalBytes));
            // 初期化（任意。巨大なら省略可）
            Unsafe.InitBlockUnaligned((void*)_buf, 0, (uint)Math.Min(totalBytes, int.MaxValue));
            _head = 0;
        }

        public void Dispose()
        {
            if (_disposed) return;
            Marshal.FreeHGlobal(_buf);
            _buf = IntPtr.Zero;
            _disposed = true;
        }

        /// <summary>
        /// 2048xH のブロックを先頭から順に追記します。容量を超える場合は false を返します。
        /// </summary>
        /// <param name="src">ブロック先頭行のポインタ</param>
        /// <param name="rows">ブロック行数（例: 512）</param>
        /// <param name="strideBytes">入力1行のバイト数（推奨: RowBytes）</param>
        /// <returns>書き込めたら true、容量超過で未書込なら false</returns>
        public bool PushBlock(IntPtr src, int rows, int strideBytes)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(LineStore));
            if (rows <= 0) return true;
            if (strideBytes < RowBytes) throw new ArgumentException("stride too small", nameof(strideBytes));
            if (src == IntPtr.Zero) throw new ArgumentNullException(nameof(src));

            long start = _head;
            long end = start + rows;
            if (end > CapacityLines) return false; // wrap しない運用

            byte* dstBase = (byte*)_buf + start * RowBytes;
            byte* srcBase = (byte*)src;

            // ★ 行ピッチが一致するなら一括コピー（最速）
            if (strideBytes == RowBytes)
            {
                long bytes = (long)rows * RowBytes;
                Buffer.MemoryCopy(srcBase, dstBase, bytes, bytes);
            }
            else
            {
                // パディングありの場合のみ行ごとコピー
                for (int i = 0; i < rows; i++)
                {
                    Buffer.MemoryCopy(
                        srcBase + (long)i * strideBytes,
                        dstBase + (long)i * RowBytes,
                        RowBytes, RowBytes);
                }
            }

            Interlocked.Exchange(ref _head, end);
            return true;
        }

        /// <summary>
        /// これまでに書いた総行数（先頭からの実データ行数）を返します。
        /// </summary>
        public long HeadLines => Interlocked.Read(ref _head);

        /// <summary>
        /// 最新 winH 行のうち、横 [x0, x0+winW) のウィンドウの先頭アドレスと行ストライドを返します（ゼロコピー）。
        /// 戻り値 true のとき、ptr は該当ウィンドウの左上画素を指し、strideBytes は 2048×elem です。
        /// </summary>
        /// <param name="winW">ウィンドウ幅（例: 725）</param>
        /// <param name="winH">ウィンドウ高さ（例: 725）</param>
        /// <param name="x0">横開始位置（0..Width-winW）</param>
        /// <param name="ptr">出力: 先頭画素ポインタ（ゼロコピー）</param>
        /// <param name="strideBytes">出力: 行ストライド（= RowBytes）</param>
        public bool TryGetLatestWindowPtr(int winW, int winH, int x0, out IntPtr ptr, out int strideBytes)
        {
            ptr = IntPtr.Zero;
            strideBytes = RowBytes;

            if (_disposed) throw new ObjectDisposedException(nameof(LineStore));
            if (winW <= 0 || winH <= 0) return false;
            if (winW > Width) return false;

            long h = HeadLines;
            if (h < winH) return false; // まだ行が足りない

            int x0Clamped = Math.Clamp(x0, 0, Math.Max(0, Width - winW));
            long topRow = h - winH;

            // wrap なし設計なので、最新 winH 行は常に連続領域。
            long byteOffset = topRow * RowBytes + (long)x0Clamped * ElemSizeBytes;
            ptr = (IntPtr)((byte*)_buf + byteOffset);
            return true;
        }
    }
}