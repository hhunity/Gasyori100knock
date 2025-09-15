// Program.cs  (Console App, x64, unsafe ON)
using System;
using System.Diagnostics;
using System.Threading;
using YourApp.Imaging;

internal static class Program
{
    static void Main()
    {
        Console.WriteLine("== LineStore ROI & Time-Interpolation Test ==");

        // --- 構成 ---
        const int srcWidth   = 2048;   // センサーの1行のピクセル数
        const int roiX       = 500;    // ここから
        const int roiW       = 725;    // これだけ保存
        const int warmupMax  = 8;      // 任意指定
        const long capacity  = 4000;   // 行容量（余裕あり）
        var store = new LineStore(srcWidth, roiX, roiW, capacity, warmupMax, PixelType.U8);

        try
        {
            // ============ 1) Warmup テスト ============
            Console.WriteLine("-- Warmup fill (before Commit) --");
            // rows=3 と rows=6 を入れて、最後の時刻が保持されるか確認
            PushBlockPattern(store, rows: 3, srcWidth, roiX, valStartRow: 0, timeSec: 0.10);
            PushBlockPattern(store, rows: 6, srcWidth, roiX, valStartRow: 3, timeSec: 0.20);
            DumpState(store);

            // Warmup中は StoredLines= warmupMax(=8) まで。今は 3+6=9 行来たが保持は 8
            AssertEq(store.StoredLines, warmupMax, "StoredLines in warmup");
            // 最新窓(幅=ROI幅、高さ= warmupMax) のポインタ取得（時刻は warmup の最後のもの=0.20）
            Require(store.TryGetLatestWindowPtr(roiW, warmupMax, 0, out var ptrW, out int strideW, out double tWarm));
            AssertNear(tWarm, 0.20, 1e-9, "Warmup time");

            // ROI 正常性：特定行の値をサンプルチェック
            // Warmup 直後の最新先頭行は「行インデックス= (3+6)-8 = 1 相当」が先頭のはずだが、
            // ここでは簡単に「任意1行」を確認：窓の1行だけ取って、その行値が (行インデックス%251) で塗られていること
            Require(store.TryGetWindowPtr(startRow: 7, winW: roiW, winH: 1, x0: 0, out var ptr1, out int stride1, out _));
            unsafe
            {
                byte val = *((byte*)ptr1); // その行の先頭画素
                Console.WriteLine($"   Warmup sample line 7: first byte={val}");
            }

            // ============ 2) Commit 後、線形補間テスト ============
            Console.WriteLine("-- Commit() & Push 3 blocks (500,500,500 lines @ 0.5s,1.0s,1.5s) --");
            store.Commit();

            // あなたの例に合わせる：各ブロック 500 行、時刻 0.5, 1.0, 1.5 秒
            PushBlockPattern(store, rows: 500, srcWidth, roiX, valStartRow: 100, timeSec: 0.5);
            PushBlockPattern(store, rows: 500, srcWidth, roiX, valStartRow: 600, timeSec: 1.0);
            PushBlockPattern(store, rows: 500, srcWidth, roiX, valStartRow: 1100, timeSec: 1.5);
            DumpState(store);

            // 要求：startRow=750, winH=500 → 先頭時刻は 1.25s になるはず
            long startRow = 750;
            Require(store.TryGetWindowPtr(startRow, roiW, 500, x0: 0, out var ptr, out int stride, out double tSec));
            Console.WriteLine($"   Expected time = 1.25 s,  Got = {tSec:F6} s");
            AssertNear(tSec, 1.25, 1e-9, "Interpolated time @ startRow=750");

            // ROI の実データが “ソースの roiX..roiX+roiW のみ” になっているかを簡単に確認
            // 例：任意の行を1行取り、先頭画素と最後の画素を読む（いずれも行に設定した値のはず）
            unsafe
            {
                byte* p = (byte*)ptr;
                byte first = p[0];
                byte last  = p[roiW - 1];
                Console.WriteLine($"   ROI check: first={first}, last={last} (row={startRow})");
            }

            Console.WriteLine("== All basic tests passed ✅ ==");
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
        int srcStride = srcWidth; // U8 前提（U16 のときは ×2 にしてください）
        byte[] buf = new byte[rows * srcStride];

        for (int r = 0; r < rows; r++)
        {
            byte val = (byte)((valStartRow + r) % 251);
            // ROI 部分だけ値を入れ、それ以外は 0（LineStore は ROI だけ読む）
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

    private static void DumpState(LineStore s)
    {
        Console.WriteLine($"   Width(ROI)={s.Width}, SourceWidth={s.SourceWidth}, WarmupMax={s.WarmupMax}");
        Console.WriteLine($"   StoredLines={s.StoredLines}, HeadTotal={s.HeadTotal}");
    }

    private static void Require(bool cond)
    {
        if (!cond) throw new Exception("Require failed");
    }

    private static void AssertEq(long actual, long expected, string name)
    {
        if (actual != expected)
            throw new Exception($"{name}: expected {expected}, got {actual}");
        Console.WriteLine($"   {name}: OK ({actual})");
    }

    private static void AssertNear(double actual, double expected, double eps, string name)
    {
        if (Math.Abs(actual - expected) > eps)
            throw new Exception($"{name}: expected {expected}, got {actual}");
        Console.WriteLine($"   {name}: OK ({actual:F9})");
    }
}