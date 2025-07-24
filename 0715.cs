extern "C" {
#include <tiffio.h>
}
#include <iostream>
#include <vector>

int main() {
    const int width = 256;
    const int height = 256;
    const char* filename = "output_2bit.tif";

    // 1ピクセル2bit → 4ピクセル = 1バイト
    const int pixels_per_byte = 4;
    const int row_bytes = (width + pixels_per_byte - 1) / pixels_per_byte;

    // データ格納バッファ（1行ごとにrow_bytes）
    std::vector<uint8_t> image(height * row_bytes, 0);

    // 画像にグラデーションを入れる（左: 0 → 右: 3）
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint8_t val2bit = (x * 4) / width; // 0〜3
            int byte_index = y * row_bytes + x / 4;
            int shift = (3 - (x % 4)) * 2;
            image[byte_index] |= (val2bit << shift);
        }
    }

    TIFF* tif = TIFFOpen(filename, "w");
    if (!tif) {
        std::cerr << "TIFFOpen failed!" << std::endl;
        return 1;
    }

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 2);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, 1);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);

    // 各行ごとに書き込み
    for (int y = 0; y < height; ++y) {
        const uint8_t* row = &image[y * row_bytes];
        if (TIFFWriteScanline(tif, (tdata_t)row, y, 0) < 0) {
            std::cerr << "Failed to write row " << y << std::endl;
            break;
        }
    }

    TIFFClose(tif);
    std::cout << "Saved: " << filename << std::endl;
    return 0;
}



extern "C" {
#include <tiffio.h>
}

int main() {
    TIFF* tif = TIFFOpen("example.tif", "r");
    if (tif) {
        uint32 width, height;
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
        std::cout << "Width: " << width << ", Height: " << height << std::endl;

        TIFFClose(tif);
    }
    return 0;
}


<UserControl x:Class="YourNamespace.BusyOverlay"
             xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
             xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
             Width="Auto" Height="Auto">
    <Grid Background="#80000000">
        <StackPanel HorizontalAlignment="Center" VerticalAlignment="Center">
            <TextBlock x:Name="messageText"
                       Text="処理中..."
                       Foreground="White"
                       FontSize="16"
                       Margin="0 0 0 10"/>
            <ProgressBar IsIndeterminate="True" Width="200" Height="20"/>
        </StackPanel>
    </Grid>
</UserControl>

public static class BusyOverlayManager
{
    private static BusyOverlay _overlay;
    private static Grid _rootGrid;

    public static void Initialize(Grid rootGrid)
    {
        _rootGrid = rootGrid;
    }

    public static void Show(string message = "処理中...")
    {
        if (_overlay == null)
        {
            _overlay = new BusyOverlay();
            Panel.SetZIndex(_overlay, 999);
            _overlay.HorizontalAlignment = HorizontalAlignment.Stretch;
            _overlay.VerticalAlignment = VerticalAlignment.Stretch;
        }

        (_overlay.FindName("messageText") as TextBlock).Text = message;

        if (!_rootGrid.Children.Contains(_overlay))
        {
            _rootGrid.Children.Add(_overlay);
        }
    }

    public static void Hide()
    {
        if (_overlay != null && _rootGrid.Children.Contains(_overlay))
        {
            _rootGrid.Children.Remove(_overlay);
        }
    }
}

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        BusyOverlayManager.Initialize(RootGrid); // ← 最初に渡すだけ
    }
}

BusyOverlayManager.Show("画像処理中...");
await Task.Run(() => HeavyProcess());
BusyOverlayManager.Hide();











private BusyOverlay _busyOverlay;

private async void StartProcess_Click(object sender, RoutedEventArgs e)
{
    // BusyOverlay をコードで生成
    _busyOverlay = new BusyOverlay
    {
        HorizontalAlignment = HorizontalAlignment.Stretch,
        VerticalAlignment = VerticalAlignment.Stretch,
        Visibility = Visibility.Visible
    };

    // 最前面に追加
    Panel.SetZIndex(_busyOverlay, 99);
    RootGrid.Children.Add(_busyOverlay);

    // UI操作をブロック（任意）
    RootGrid.IsEnabled = false;

    try
    {
        await Task.Run(() =>
        {
            Thread.Sleep(3000); // 重たい処理
        });
    }
    finally
    {
        // 終了時に削除
        RootGrid.Children.Remove(_busyOverlay);
        RootGrid.IsEnabled = true;
    }
}

<Window ...
    xmlns:local="clr-namespace:YourNamespace">

    <Grid x:Name="RootGrid">
        <!-- 画面UI -->
        <StackPanel>
            <Button Content="処理開始" Click="StartProcess_Click"/>
        </StackPanel>

        <!-- オーバーレイ -->
        <local:BusyOverlay x:Name="busyOverlay" Visibility="Collapsed"
                           Panel.ZIndex="99"/>
    </Grid>
</Window>

private async void StartProcess_Click(object sender, RoutedEventArgs e)
{
    busyOverlay.Visibility = Visibility.Visible;
    RootGrid.IsEnabled = false;

    try
    {
        await Task.Run(() => {
            Thread.Sleep(3000); // 重い処理
        });
    }
    finally
    {
        busyOverlay.Visibility = Visibility.Collapsed;
        RootGrid.IsEnabled = true;
    }
}

<UserControl x:Class="YourNamespace.BusyOverlay"
             xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
             xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
             Visibility="Collapsed"
             Background="#80000000">
    <Grid>
        <StackPanel HorizontalAlignment="Center" VerticalAlignment="Center">
            <TextBlock Text="処理中..." Foreground="White" FontSize="16" Margin="0 0 0 10"/>
            <ProgressBar IsIndeterminate="True" Width="200" Height="20"/>
        </StackPanel>
    </Grid>
</UserControl>





static void RegisterCustomTag()
{
    Tiff.TagExtender extender = new Tiff.TagExtender(CustomTagExtender);
    Tiff.SetTagExtender(extender);
}

private static void CustomTagExtender(Tiff tif)
{
    TiffFieldInfo[] tiffFieldInfo = {
        new TiffFieldInfo(MyCustomTag, -1, -1, TiffType.UNDEFINED, FieldBit.Custom, true, false, "MyCustomTag")
    };
    tif.MergeFieldInfo(tiffFieldInfo, tiffFieldInfo.Length);
}

RegisterCustomTag(); // カスタムタグ登録を忘れずに！

using (Tiff tif = Tiff.Open("output.tif", "w"))
{
    int width = 256;
    int height = 256;
    tif.SetField(TiffTag.IMAGEWIDTH, width);
    tif.SetField(TiffTag.IMAGELENGTH, height);
    tif.SetField(TiffTag.BITSPERSAMPLE, 8);
    tif.SetField(TiffTag.SAMPLESPERPIXEL, 1);
    tif.SetField(TiffTag.ROWSPERSTRIP, height);
    tif.SetField(TiffTag.COMPRESSION, Compression.NONE);
    tif.SetField(TiffTag.PHOTOMETRIC, Photometric.MINISBLACK);
    tif.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG);

    // カスタムタグ書き込み
    byte[] customData = new byte[1024];
    new Random().NextBytes(customData);
    tif.SetField(MyCustomTag, customData.Length, customData);

    // 画像データ（ダミー）
    byte[] buffer = new byte[width];
    for (int i = 0; i < height; i++)
        tif.WriteScanline(buffer, i);
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