// Program.cs  (Console App, x64, unsafe=ON)
using System;
using YourApp.Imaging;

internal static class Program
{
    static void Main()
    {
        Console.WriteLine("== LineStore ROI + Warmup per-line + Interpolation test ==");

        const int srcWidth  = 2048;
        const int roiX      = 500;
        const int roiW      = 725;
        const int warmupMax = 500;     // ★ ウォームアップ 500
        const long capacity = 4000;

        var store = new LineStore(srcWidth, roiX, roiW, capacity, warmupMax, PixelType.U8);

        try
        {
            // =========================
            // 1) ウォームアップ per-line 時刻の検証
            // =========================
            Console.WriteLine("-- Warmup per-line timestamps --");
            // 300 @ 0.10s → 300 @ 0.20s を push。
            // WarmupMax=500 なので保持されるのは最後の500行＝[100..599]。
            // よって 0..199 が 0.10s、200..499 が 0.20s に対応。
            PushBlockPattern(store, rows: 300, srcWidth, roiX, valStartRow: 0,   timeSec: 0.10);
            PushBlockPattern(store, rows: 300, srcWidth, roiX, valStartRow: 300, timeSec: 0.20);

            // 行150 → 0.10s（前半側）
            Require(store.TryGetWindowPtr(150, roiW, 1, 0, out _, out _, out double t150));
            AssertNear(t150, 0.10, 1e-9, "Warmup per-line time @ row 150");

            // 行350 → 0.20s（後半側）
            Require(store.TryGetWindowPtr(350, roiW, 1, 0, out var p350, out int stride350, out double t350));
            AssertNear(t350, 0.20, 1e-9, "Warmup per-line time @ row 350");

            // ROIの簡易チェック（行350の先頭・末尾ピクセル）
            unsafe
            {
                byte vFirst = *((byte*)p350 + 0);
                byte vLast  = *((byte*)p350 + (roiW - 1));
                Console.WriteLine($"   ROI sample row350 -> first={vFirst}, last={vLast}");
            }

            // =========================
            // 2) Commit 後の線形補間（絶対行指定）検証
            // =========================
            Console.WriteLine("-- Commit & interpolation (absolute rows) --");
            store.Commit();

            // 各500行のブロックを 0.5s / 1.0s / 1.5s で追加
            PushBlockPattern(store, rows: 500, srcWidth, roiX, valStartRow: 600,  timeSec: 0.5);
            PushBlockPattern(store, rows: 500, srcWidth, roiX, valStartRow: 1100, timeSec: 1.0);
            PushBlockPattern(store, rows: 500, srcWidth, roiX, valStartRow: 1600, timeSec: 1.5);

            // ★注意: Warmup=500 なので、絶対行でのブロック境界は 500/1000/1500…
            // a) startRow=750（1stブロックの中間）→ 0.75s が正解
            Require(store.TryGetWindowPtr(750, roiW, 500, 0, out var pA, out int strideA, out double tA));
            Console.WriteLine($"   Expected 0.75 s @ abs 750, Got = {tA:F6} s");
            AssertNear(tA, 0.75, 1e-9, "Interpolated time @ abs row 750");

            // b) startRow=1250（2ndブロックの中間）→ 1.25s が正解
            Require(store.TryGetWindowPtr(1250, roiW, 500, 0, out var pB, out int strideB, out double tB));
            Console.WriteLine($"   Expected 1.25 s @ abs 1250, Got = {tB:F6} s");
            AssertNear(tB, 1.25, 1e-9, "Interpolated time @ abs row 1250");

            Console.WriteLine("== All tests passed ✅ ==");
        }
        finally
        {
            store.Dispose();
        }
    }

    // ----------------- ヘルパ -----------------

    // rows 行のブロックを Push：各行の ROI 部分は (valStartRow + r) % 251 の定数で塗る
    private static unsafe void PushBlockPattern(LineStore store, int rows, int srcWidth, int roiX, int valStartRow, double timeSec)
    {
        if (store.PixelType != PixelType.U8)
            throw new NotSupportedException("Test assumes PixelType.U8 (1 byte/pixel)");

        int srcStride = srcWidth; // U8
        byte[] buf = new byte[rows * srcStride];

        for (int r = 0; r < rows; r++)
        {
            byte val = (byte)((valStartRow + r) % 251);
            int start = r * srcStride + roiX;
            for (int x = 0; x < store.Width; x++)
                buf[start + x] = val;
        }

        fixed (byte* p = buf)
        {
            bool ok = store.PushBlock((IntPtr)p, rows, srcStride, timeSec);
            if (!ok) throw new Exception("PushBlock returned false (capacity exhausted)");
        }
    }

    private static void Require(bool cond)
    {
        if (!cond) throw new Exception("Require failed");
    }

    private static void AssertNear(double actual, double expected, double eps, string name)
    {
        if (Math.Abs(actual - expected) > eps)
            throw new Exception($"{name}: expected {expected}, got {actual}");
        Console.WriteLine($"   {name}: OK ({actual:F9})");
    }
}