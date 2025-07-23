
public enum MyCustomTags
{
    MyTag = 65000
}

TiffFieldInfo[] infos = {
    new TiffFieldInfo((TiffTag)MyCustomTags.MyTag, -1, -1, TiffType.UNDEFINED, FieldBit.Custom, true, false, "MyTag")
};


using (Tiff output = Tiff.Open("output.tif", "w"))
{
    // 画像の基本情報を設定
    output.SetField(TiffTag.IMAGEWIDTH, 256);
    output.SetField(TiffTag.IMAGELENGTH, 256);
    output.SetField(TiffTag.SAMPLESPERPIXEL, 1);
    output.SetField(TiffTag.BITSPERSAMPLE, 8);
    output.SetField(TiffTag.ROWSPERSTRIP, 256);
    output.SetField(TiffTag.COMPRESSION, Compression.NONE);
    output.SetField(TiffTag.PHOTOMETRIC, Photometric.MINISBLACK);

    // カスタムタグ情報を登録
    output.SetField(TiffTag.EXTENSION, infos); // ←重要
    byte[] myData = new byte[1024]; // 例：1KBの独自データ
    new Random().NextBytes(myData); // ダミーデータを格納

    // カスタムタグ書き込み
    output.SetField((TiffTag)MyCustomTags.MyTag, myData.Length, myData);

    // ダミーの画像データを書き込む（全黒）
    byte[] scanline = new byte[256];
    for (int i = 0; i < 256; i++)
    {
        output.WriteScanline(scanline, i);
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