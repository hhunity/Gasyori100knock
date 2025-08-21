using System;
using System.Diagnostics;

public sealed class SyncIJ
{
    double betaSec;                      // カメラ0秒のPC時刻[sec]
    readonly double hostHz = Stopwatch.Frequency;

    public interface INodeMap
    {
        void Execute(string cmd);        // "TimestampReset"
    }

    // 起動ごと（印刷ごと）に実行
    public void Start(INodeMap nm)
    {
        long t0 = Stopwatch.GetTimestamp();
        nm.Execute("TimestampReset");    // カメラ時刻を 0ns に
        long t1 = Stopwatch.GetTimestamp();
        betaSec = (t0 + (t1 - t0)/2.0) / hostHz;   // 片道補正したPC秒
    }

    // 露光Chunk(ns) → PCのStopwatch tickへ
    public long CamNsToHostTicks(ulong camNs)
    {
        double tHostSec = camNs * 1e-9 + betaSec;
        return (long)Math.Round(tHostSec * hostHz);
    }

double alpha = 1.0; // ドリフト補正係数

public void MidpointAdjust(double camMidNs)
{
    // 中間で: long a=sw(); latch(); read camMidNs; long b=sw();
    // ここでは呼び元で camMidNs を取得済みとする
    long a = /* 送信直前 sw */ 0;
    long b = /* 応答直後 sw */ 0;
    double tHostMid = (a + (b - a)/2.0) / hostHz;
    alpha = (tHostMid - betaSec) / (camMidNs * 1e-9);
}

public long CamNsToHostTicksAlpha(ulong camNs)
{
    double tHostSec = alpha * (camNs * 1e-9) + betaSec;
    return (long)Math.Round(tHostSec * hostHz);
}
}