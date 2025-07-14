

using System;
using PvDotNet;

class Program
{
    static void Main(string[] args)
    {
        using (PvSystem system = new PvSystem())
        using (PvDevice device = PvDevice.CreateAndConnect())
        using (PvStream stream = PvStream.CreateAndOpen(device.ConnectionID))
        {
            // ✅ チャンクタイムスタンプを有効化（GenICam）
            EnableChunkTimestamp(device);

            // バッファ準備
            const int bufferCount = 20;
            PvBuffer[] buffers = new PvBuffer[bufferCount];
            for (int i = 0; i < bufferCount; ++i)
            {
                buffers[i] = new PvBuffer();
                stream.QueueBuffer(buffers[i]);
            }

            device.StreamEnable();
            Console.WriteLine("受信開始（チャンクタイムスタンプ有効）。");

            ulong lastBlockID = 0;
            ulong lastTimestamp = 0;

            while (true)
            {
                PvBuffer buffer = null;
                PvResult result = stream.RetrieveBuffer(ref buffer, 1000);

                if (!result.IsOK)
                {
                    Console.WriteLine($"[❌] RetrieveBuffer error: {result.CodeString}");
                    continue;
                }

                if (buffer.Status != PvBufferStatus.Ok)
                {
                    Console.WriteLine($"[⚠] Buffer Status: {buffer.Status}");

                    if (buffer.Status == PvBufferStatus.Incomplete)
                        Console.WriteLine($"  ↳ Missing packets: {buffer.MissingPacketCount}");

                    if (buffer.Status == PvBufferStatus.AutoAbort)
                        Console.WriteLine("  ↳ AutoAbort detected");
                }

                ulong blockID = buffer.BlockID;
                if (lastBlockID != 0 && blockID != lastBlockID + 1)
                    Console.WriteLine($"[⚠] Frame skipped: {lastBlockID} → {blockID}");
                lastBlockID = blockID;

                // ✅ Chunk Timestamp 取得
                if (buffer.ChunkEnabled)
                {
                    ulong chunkTimestamp = buffer.ChunkTimestamp;
                    if (lastTimestamp != 0)
                    {
                        ulong delta = chunkTimestamp - lastTimestamp;
                        Console.WriteLine($"[⏱] ChunkTimestamp Interval: {delta} ticks");
                    }
                    lastTimestamp = chunkTimestamp;

                    Console.WriteLine($"[✔] Frame {blockID}, Timestamp = {chunkTimestamp}");
                }
                else
                {
                    Console.WriteLine($"[⚠] Chunk data not enabled in this buffer.");
                }

                stream.QueueBuffer(buffer);
            }
        }
    }

    static void EnableChunkTimestamp(PvDevice device)
    {
        PvGenParameterArray parameters = device.Parameters;

        Console.WriteLine("チャンクタイムスタンプ設定中...");

        // ✅ ChunkMode 有効化
        parameters["ChunkModeActive"].SetValue(true);

        // ✅ Timestamp チャンクを有効にする
        parameters["ChunkSelector"].SetValue("Timestamp");
        parameters["ChunkEnable"].SetValue(true);

        Console.WriteLine("ChunkTimestamp 有効化完了");
    }
}