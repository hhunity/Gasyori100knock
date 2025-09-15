using System;
using System.Runtime.InteropServices;
using System.Threading;

namespace YourApp.Imaging
{
  public enum PixelType { U8 = 0, U16 = 1 }

  /// <summary>
  /// Line sensor buffer:
  ///  - Warmup: keep latest up to 6 lines (contiguous at 0..5). When over, shift-left and append.
  ///  - Commit(): from the next push, linearly append from index=6 up to CapacityLines.
  /// Concurrency: 1 writer (PushBlock) + N readers (TryGet...); lock-free publication with Volatile.
  /// </summary>
  public unsafe sealed class LineStore : IDisposable
  {
    public int Width { get; }
    public long CapacityLines { get; }
    public PixelType PixelType { get; }
    public int ElemSizeBytes { get; }
    public int RowBytes => Width * ElemSizeBytes;

    // Warmup control
    private const int WARMUP_MAX = 6;
    private volatile bool _committed;    // false: warmup (0..6), true: linear append
    private int _warmupCount;            // 0..6 (warmup only)

    // Buffer & indices
    private IntPtr _buf;                 // CapacityLines x RowBytes
    private long _writeIndex;          // next write line (commit後: starts from warmupCount)
    private long _headTotal;           // total successfully received lines (monotonic)
    private long _storedLines;         // published usable contiguous line count (0..6 before, then 6..CapacityLines)

    private volatile bool _disposed;

    public LineStore(int width, long capacityLines, PixelType pt)
    {
      if (IntPtr.Size == 4) throw new PlatformNotSupportedException("Use x64.");
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

      _committed = false;
      _warmupCount = 0;
      _writeIndex = 0;
      Volatile.Write(ref _storedLines, 0);
      Interlocked.Exchange(ref _headTotal, 0);
      Volatile.Write(ref _disposed, false);
    }

    public void Dispose()
    {
      if (Volatile.Read(ref _disposed)) return;

      // 書き込み/参照と競合しないよう、最後に公開フラグを立てる
      Marshal.FreeHGlobal(_buf);
      _buf = IntPtr.Zero;

      Volatile.Write(ref _disposed, true);
    }

    /// <summary>
    /// End warmup; subsequent PushBlock appends linearly starting at index = warmupCount (≤6).
    /// </summary>
    public void Commit()
    {
      if (Volatile.Read(ref _disposed)) throw new ObjectDisposedException(nameof(LineStore));
      if (Volatile.Read(ref _committed)) return;

      // Warmupでは最新行群が 0.._warmupCount-1 に時系列順で常に配置済み
      _writeIndex = _warmupCount;
      // 公開（可視性: 先に値を置き、最後にフラグ）
      Volatile.Write(ref _storedLines, _warmupCount);
      Volatile.Write(ref _committed, true);
    }

    /// <summary>
    /// Ingest a block of rows. Warmup: maintain up to 6 lines contiguous at head.
    /// After Commit: append linearly until capacity. Returns false if capacity is exhausted (post-commit).
    /// </summary>
    public bool PushBlock(IntPtr src, int rows, int strideBytes)
    {
      if (Volatile.Read(ref _disposed)) throw new ObjectDisposedException(nameof(LineStore));
      if (src == IntPtr.Zero) throw new ArgumentNullException(nameof(src));
      if (rows <= 0) return true;
      if (strideBytes < RowBytes) throw new ArgumentException("stride too small");

      if (!Volatile.Read(ref _committed))
      {
        PushWarmup(src, rows, strideBytes);
        Interlocked.Add(ref _headTotal, rows);
        return true;
      }
      else
      {
        return PushLinear(src, rows, strideBytes);
      }
    }

    // ---- Warmup: at most 6 lines kept contiguous at 0..5. Block-wise O(6) copies, overlap-safe. ----
    private void PushWarmup(IntPtr src, int rows, int strideBytes)
    {
      byte* sBase = (byte*)src;
      byte* dBase = (byte*)_buf;

      int filled = _warmupCount; // 0..6

      // 1) Fill until full (≤6)
      if (filled < WARMUP_MAX)
      {
        int need = WARMUP_MAX - filled;
        int take = rows < need ? rows : need;

        if (strideBytes == RowBytes)
        {
          long bytes = (long)take * RowBytes;
          Buffer.MemoryCopy(sBase, dBase + (long)filled * RowBytes, bytes, bytes);
        }
        else
        {
          for (int i = 0; i < take; i++)
          {
            byte* srcLine = sBase + (long)i * strideBytes;
            byte* dstLine = dBase + (long)(filled + i) * RowBytes;
            Buffer.MemoryCopy(srcLine, dstLine, RowBytes, RowBytes);
          }
        }

        filled += take;
        _warmupCount = filled;
        Volatile.Write(ref _storedLines, filled); // publish newly available lines

        rows -= take;
        sBase += (long)take * strideBytes;

        if (rows <= 0) return; // not yet full or just full now
      }

      // 2) Now filled == 6; update "latest 6" from incoming block
      if (rows >= WARMUP_MAX)
      {
        // Replace with the tail 6 lines of this block
        byte* tail = sBase + (long)(rows - WARMUP_MAX) * strideBytes;
        if (strideBytes == RowBytes)
        {
          long bytes = (long)WARMUP_MAX * RowBytes;
          Buffer.MemoryCopy(tail, dBase, bytes, bytes);
        }
        else
        {
          for (int i = 0; i < WARMUP_MAX; i++)
          {
            Buffer.MemoryCopy(
                tail + (long)i * strideBytes,
                dBase + (long)i * RowBytes,
                RowBytes, RowBytes);
          }
        }
        // storedLines remains 6
        Volatile.Write(ref _storedLines, WARMUP_MAX);
      }
      else // 0 < rows < 6
      {
        int shift = rows;                    // left-shift by rows
        int keep = WARMUP_MAX - shift;      // keep first 'keep' lines after shift

        // 2-a) shift-left existing 6 lines by 'shift' (overlap-safe: low to high)
        for (int y = 0; y < keep; y++)
        {
          byte* srcRow = dBase + (long)(y + shift) * RowBytes;
          byte* dstRow = dBase + (long)y * RowBytes;
          Buffer.MemoryCopy(srcRow, dstRow, RowBytes, RowBytes);
        }

        // 2-b) append new rows to the tail
        for (int i = 0; i < rows; i++)
        {
          byte* srcLine = sBase + (long)i * strideBytes;
          byte* dstLine = dBase + (long)(keep + i) * RowBytes;
          Buffer.MemoryCopy(srcLine, dstLine, RowBytes, RowBytes);
        }

        Volatile.Write(ref _storedLines, WARMUP_MAX);
      }
    }

    // ---- After Commit: linear append until capacity. ----
    private bool PushLinear(IntPtr src, int rows, int strideBytes)
    {
      long remain = CapacityLines - _writeIndex;
      if (remain <= 0) return false; // full

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
          Buffer.MemoryCopy(srcLine, dstLine, RowBytes, RowBytes);
        }
      }

      Interlocked.Add(ref _headTotal, can);
      _writeIndex += can;

      // Publish newly available lines (Release); readers will Acquire via Volatile.Read
      long newStored = _writeIndex;
      if (newStored > Volatile.Read(ref _storedLines))
        Volatile.Write(ref _storedLines, newStored);

      return can == rows; // false if we hit capacity mid-block
    }

    // --- Info / Properties ---
    public long HeadTotal => Interlocked.Read(ref _headTotal);  // statistics (monotonic)
    public long StoredLines => Volatile.Read(ref _storedLines);   // usable contiguous lines now

    /// <summary>
    /// Latest window (winH) with width winW at x0. Returns pointer & stride (RowBytes).
    /// </summary>
    public bool TryGetLatestWindowPtr(int winW, int winH, int x0, out IntPtr ptr, out int strideBytes)
    {
      if (Volatile.Read(ref _disposed)) throw new ObjectDisposedException(nameof(LineStore));
      ptr = IntPtr.Zero; strideBytes = RowBytes;

      if (winW <= 0 || winH <= 0 || winW > Width) return false;

      long avail = Volatile.Read(ref _storedLines);
      if (avail < winH) return false;

      long topRow = avail - winH;
      return TryGetWindowPtr(topRow, winW, winH, x0, out ptr, out strideBytes);
    }

    /// <summary>
    /// Window from arbitrary startRow (0-based) with height winH and width winW at x0.
    /// </summary>
    public bool TryGetWindowPtr(long startRow, int winW, int winH, int x0,
                                out IntPtr ptr, out int strideBytes)
    {
      if (Volatile.Read(ref _disposed)) throw new ObjectDisposedException(nameof(LineStore));
      ptr = IntPtr.Zero; strideBytes = RowBytes;

      if (startRow < 0 || winW <= 0 || winH <= 0 || winW > Width) return false;

      long avail = Volatile.Read(ref _storedLines);
      if (startRow + winH > avail) return false;

      int x0c = Clamp(x0, 0, Math.Max(0, Width - winW));
      long byteOffset = startRow * (long)RowBytes + (long)x0c * ElemSizeBytes;
      ptr = (IntPtr)((byte*)_buf + byteOffset);
      return true;
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static int Clamp(int v, int min, int max)
    {
      if (v < min) return min;
      if (v > max) return max;
      return v;
    }
  }

  class Program
  {
    static void Main(string[] args)
    {
      Console.WriteLine("Hello World!");
    }
  }
}

