
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
            // バッファの準備
            const int bufferCount = 20;
            PvBuffer[] buffers = new PvBuffer[bufferCount];
            for (int i = 0; i < bufferCount; ++i)
            {
                buffers[i] = new PvBuffer();
                stream.QueueBuffer(buffers[i]);
            }

            device.StreamEnable();
            Console.WriteLine("受信開始（Status + OperationalResult 確認）");

            while (true)
            {
                PvBuffer buffer = null;
                PvResult result = stream.RetrieveBuffer(ref buffer, 1000);

                if (!result.IsOK)
                {
                    Console.WriteLine($"[❌] RetrieveBuffer failed: {result.OperationalResult}");
                    continue;
                }

                // Buffer のステータスとオペレーション結果を出力
                Console.WriteLine($"[✔] Frame BlockID = {buffer.BlockID}");
                Console.WriteLine($"     Status: {buffer.Status}");
                Console.WriteLine($"     OperationalResult: {buffer.OperationalResult}");

                // 警告条件のチェック
                if (buffer.Status != PvBufferStatus.Ok)
                {
                    Console.WriteLine($"[⚠] BufferStatus = {buffer.Status}");

                    if (buffer.Status == PvBufferStatus.Incomplete)
                        Console.WriteLine($"     ↳ Missing packets: {buffer.MissingPacketCount}");

                    if (buffer.Status == PvBufferStatus.AutoAbort)
                        Console.WriteLine("     ↳ AutoAbort: バッファ不足やRetrieve遅延の可能性");
                }

                if (buffer.OperationalResult != "OK")
                {
                    Console.WriteLine($"[⚠] 非正常な OperationalResult: {buffer.OperationalResult}");
                }

                // バッファを戻す
                stream.QueueBuffer(buffer);
            }
        }
    }
}

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
            // バッファの準備
            const int bufferCount = 20;
            PvBuffer[] buffers = new PvBuffer[bufferCount];
            for (int i = 0; i < bufferCount; ++i)
            {
                buffers[i] = new PvBuffer();
                stream.QueueBuffer(buffers[i]);
            }

            device.StreamEnable();
            Console.WriteLine("受信開始（PvResult + BufferStatus ログ）");

            while (true)
            {
                PvBuffer buffer = null;
                PvResult result = stream.RetrieveBuffer(ref buffer, 1000);

                if (!result.IsOK)
                {
                    Console.WriteLine("[❌] RetrieveBuffer failed:");
                    Console.WriteLine($"      Code: {result.Code}");
                    Console.WriteLine($"      CodeString: {result.CodeString}");
                    Console.WriteLine($"      Description: {result.Description}");
                    Console.WriteLine($"      OperationalResult: {result.OperationalResult}");
                    continue;
                }

                // PvResultはOKだった → PvBufferの中身をチェック
                PvBufferStatus status = buffer.Status;

                if (status != PvBufferStatus.Ok)
                {
                    Console.WriteLine($"[⚠] Buffer Status: {status}");

                    if (status == PvBufferStatus.Incomplete)
                        Console.WriteLine($"     ↳ Missing packets: {buffer.MissingPacketCount}");

                    if (status == PvBufferStatus.AutoAbort)
                        Console.WriteLine("     ↳ AutoAbort: Queue不足 or Retrieve遅延");
                }
                else
                {
                    Console.WriteLine($"[✔] Frame OK: BlockID = {buffer.BlockID}, Size = {buffer.Image.Width} x {buffer.Image.Height}");
                }

                // バッファを戻す
                stream.QueueBuffer(buffer);
            }
        }
    }
}