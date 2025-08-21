
using System;
using System.Diagnostics;
using PvDotNet;   // eBUS .NET 名前空間に合わせて調整してください

class Program
{
    static void Main()
    {
        // ---- デバイス接続 ----
        string connectionID = "192.168.1.10"; // IPやID
        PvDevice device = PvDevice.CreateAndConnect(connectionID);
        PvGenParameterArray deviceParams = device.GetParameters();

        // ---- TimestampLatch 実行 ----
        PvGenCommand latchCmd = deviceParams.GetCommand("TimestampControlLatch");
        latchCmd.Execute();

        // ---- カメラ現在時刻（ns）取得 ----
        // 機種により "TimestampLatchValue" または "TimestampValue" になることがあります
        long camNs = 0;
        var tsVal = deviceParams.GetInteger("TimestampLatchValue");
        if (tsVal != null)
            camNs = tsVal.Value;
        else
        {
            var tsValAlt = deviceParams.GetInteger("TimestampValue");
            if (tsValAlt != null) camNs = tsValAlt.Value;
        }

        // ---- エンコーダ値取得 ----
        // EncoderSelector=Encoder1/Encoder2 を切り替えてから EncoderValue を読むのがSFNC標準
        PvGenEnum encSel = deviceParams.GetEnum("EncoderSelector");
        encSel.SetValue("Encoder1"); // 例: Encoder1
        PvGenInteger encVal = deviceParams.GetInteger("EncoderValue");
        long encoderCount = encVal?.Value ?? 0;

        // ---- PC側 Stopwatch 時刻も同時に取得 ----
        long swTicks = Stopwatch.GetTimestamp();
        double swSec = (double)swTicks / Stopwatch.Frequency;

        // ---- 表示 ----
        Console.WriteLine(
            $"Now: CamTimestamp={camNs} ns ({camNs * 1e-9:F6} s), " +
            $"Encoder={encoderCount}, PC_Stopwatch={swSec:F6} s");

        device.Disconnect();
    }
}


using System;
using System.Diagnostics;
using PvDotNet;   // 実際の SDK で指定する using に合わせてください

class Program
{
    static void Main()
    {
        // ---- カメラに接続 ----
        string connectionID = "192.168.1.10"; // 例: IPアドレスまたはデバイスID
        PvDevice device = PvDevice.CreateAndConnect(connectionID);
        PvGenParameterArray deviceParams = device.GetParameters();

        // ---- Timestamp Reset (印刷開始前に0リセット) ----
        PvGenCommand tsReset = deviceParams.GetCommand("TimestampReset");
        tsReset.Execute();

        // ---- Chunk 有効化 ----
        PvGenEnum chunkSelector = deviceParams.GetEnum("ChunkSelector");
        PvGenBoolean chunkEnable = deviceParams.GetBoolean("ChunkEnable");

        // Timestamp
        chunkSelector.SetValue("Timestamp");
        chunkEnable.SetValue(true);

        // Encoder (機種によって "EncoderValue" → "ChunkEncoderValue" の場合あり)
        chunkSelector.SetValue("EncoderValue");
        chunkEnable.SetValue(true);

        // ---- ストリーム & パイプライン準備 ----
        PvStream stream = PvStream.CreateAndOpen(connectionID);
        PvPipeline pipeline = new PvPipeline(stream);
        pipeline.Start();

        // ---- Acquisition Start ----
        PvGenCommand acqStart = deviceParams.GetCommand("AcquisitionStart");
        acqStart.Execute();

        // ---- フレーム受信ループ ----
        for (int i = 0; i < 10; i++)
        {
            PvBuffer buffer = pipeline.RetrieveNextBuffer(1000); // timeout 1000ms
            if (buffer != null && buffer.OperationResult.IsOK())
            {
                // Stopwatch 側で PC の現在時刻も取る
                long swTicks = Stopwatch.GetTimestamp();

                // Chunk データ参照
                PvGenParameterArray chunks = buffer.GetChunkData();

                long tsNs = 0;
                long encVal = 0;

                // ChunkTimestamp
                var tsParam = chunks.GetInteger("ChunkTimestamp");
                if (tsParam != null) tsNs = tsParam.Value;

                // ChunkEncoderValue
                var encParam = chunks.GetInteger("ChunkEncoderValue");
                if (encParam != null) encVal = encParam.Value;

                double tsSec = tsNs * 1e-9;
                double swSec = (double)swTicks / Stopwatch.Frequency;

                Console.WriteLine(
                    $"Frame {i:D2}: CamTimestamp={tsNs} ns ({tsSec:F6} s), " +
                    $"Encoder={encVal}, PC_Stopwatch={swSec:F6} s");
            }

            pipeline.ReleaseBuffer(buffer);
        }

        // ---- Acquisition Stop ----
        PvGenCommand acqStop = deviceParams.GetCommand("AcquisitionStop");
        acqStop.Execute();

        pipeline.Stop();
        device.Disconnect();
        stream.Close();
    }
}




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