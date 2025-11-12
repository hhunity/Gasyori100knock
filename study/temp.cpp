#include <tiffio.h>
#include <cstdint>
#include <stdexcept>

void write_tiff_gray8(const char* path,
                      uint32_t w, uint32_t h,
                      const uint8_t* data,     // 1px=1byte、行ピッチ=stride
                      uint32_t stride_bytes,   // 例: stride_bytes = w
                      int use_lzw = 1)         // 0:無圧縮, 1:LZW
{
    TIFF* tif = TIFFOpen(path, "w");
    if (!tif) throw std::runtime_error("TIFFOpen failed");

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, w);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, h);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, use_lzw ? COMPRESSION_LZW : COMPRESSION_NONE);

    // ストリップサイズ（適当でOK：1〜数十行）。小さめにするとメモリ少なめ。
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(tif, w));

    for (uint32_t y = 0; y < h; ++y) {
        const tdata_t row = (tdata_t)(data + (size_t)y * stride_bytes);
        if (TIFFWriteScanline(tif, row, y, 0) < 0)
            throw std::runtime_error("TIFFWriteScanline failed");
    }

    // お好みでメタデータ
    TIFFSetField(tif, TIFFTAG_SOFTWARE, "YourApp 1.0");
    TIFFClose(tif);
}


#include <tiffio.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>

uint8_t* read_tiff_gray8(const char* filename,
                         uint32_t& width,
                         uint32_t& height)
{
    TIFF* tif = TIFFOpen(filename, "r");
    if (!tif) {
        std::cerr << "Cannot open file\n";
        return nullptr;
    }

    uint16_t bitsPerSample = 0, samplesPerPixel = 0;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);

    if (bitsPerSample != 8 || samplesPerPixel != 1) {
        std::cerr << "Only 8bit grayscale supported\n";
        TIFFClose(tif);
        return nullptr;
    }

    // ★ ここで自前のメモリを確保
    uint8_t* buf = static_cast<uint8_t*>(std::malloc(width * height));
    if (!buf) {
        TIFFClose(tif);
        return nullptr;
    }

    for (uint32_t y = 0; y < height; ++y) {
        uint8_t* row = buf + y * width;
        if (TIFFReadScanline(tif, row, y, 0) < 0) {
            std::cerr << "Failed to read scanline " << y << "\n";
            std::free(buf);
            TIFFClose(tif);
            return nullptr;
        }
    }

    TIFFClose(tif);
    return buf;  // ← 呼び出し側で free() する
}

int main() {
    uint32_t w, h;
    uint8_t* data = read_tiff_gray8("gray8.tif", w, h);
    if (data) {
        std::cout << "read ok: " << w << "x" << h
                  << " first=" << (int)data[0] << "\n";
        std::free(data);
    }
}

#include <tiffio.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>

uint8_t* read_tiff_gray2(const char* filename,
                         uint32_t& w,
                         uint32_t& h)
{
    TIFF* tif = TIFFOpen(filename, "r");
    if (!tif) return nullptr;

    uint16_t bitsPerSample = 0, samplesPerPixel = 0;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);

    if (bitsPerSample != 2 || samplesPerPixel != 1) {
        std::cerr << "Not 2-bit grayscale\n";
        TIFFClose(tif);
        return nullptr;
    }

    const uint32_t packed_stride = (w + 3) / 4;
    std::vector<uint8_t> packed(packed_stride * h);
    for (uint32_t y = 0; y < h; ++y)
        TIFFReadScanline(tif, packed.data() + y * packed_stride, y, 0);
    TIFFClose(tif);

    // ★ 展開先メモリを確保（8bit化したいので1pix=1byte）
    uint8_t* out = static_cast<uint8_t*>(std::malloc(w * h));
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            const uint8_t byte = packed[y * packed_stride + x / 4];
            const uint8_t shift = 6 - 2 * (x % 4);
            uint8_t v2 = (byte >> shift) & 0x3; // 0〜3
            out[y * w + x] = (v2 * 255) / 3;    // 0〜255に拡張
        }
    }
    return out;
}

int main() {
    uint32_t w, h;
    uint8_t* img = read_tiff_gray2("gray2.tif", w, h);
    if (img) {
        std::cout << "size=" << w << "x" << h << " first=" << (int)img[0] << "\n";
        std::free(img);
    }
}




#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/component/component.hpp>
#include <ftxui/dom/elements.hpp>
#include <thread>
#include <atomic>
#include <vector>
#include <string>
using namespace ftxui;
using namespace std::chrono_literals;

enum class Page { Menu, Task };

int main() {
  auto screen = ScreenInteractive::TerminalOutput();
  Page page = Page::Menu;
  std::vector<std::string> logs;
  std::atomic<bool> running{false};
  int progress = 0;

  auto start_task = [&] {
    if (running) return;
    running = true;
    logs.clear();
    std::thread([&] {
      for (int i = 1; i <= 10; ++i) {
        logs.push_back("step " + std::to_string(i) + "/10 done");
        progress = i * 10;
        screen.PostEvent(Event::Custom); // 再描画
        std::this_thread::sleep_for(300ms);
      }
      running = false;
      page = Page::Menu;
      screen.PostEvent(Event::Custom);
    }).detach();
  };

  std::vector<std::string> menu_items = {"Run task", "Exit"};
  int selected = 0;
  auto menu = Menu(&menu_items, &selected);
  menu |= CatchEvent([&](Event e) {
    if (e == Event::Return) {
      if (selected == 0) { page = Page::Task; start_task(); }
      else if (selected == 1) screen.Exit();
      return true;
    }
    return false;
  });

  auto task_view = Renderer([&] {
    std::vector<Element> lines;
    for (auto& s : logs) lines.push_back(text(s));
    return vbox({
      text("Running task...") | bold,
      gauge(progress / 100.0f),
      vbox(std::move(lines)) | vscroll_indicator | yframe,
    }) | border;
  });

  auto root = Renderer([&] {
    if (page == Page::Menu)
      return window(text("Menu"), menu->Render()) | border;
    else
      return task_view->Render();
  });

  screen.Loop(root);
}


#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/component/component.hpp>
#include <ftxui/dom/elements.hpp>
#include <iostream>
using namespace ftxui;

int main() {
    auto screen = ScreenInteractive::TerminalOutput();

    std::vector<std::string> menu_entries = {
        "1. カメラ取込み開始",
        "2. 画像処理テスト",
        "3. 送信テスト",
        "4. 終了"
    };
    int selected = 0;

    auto menu = Menu(&menu_entries, &selected);
    menu |= CatchEvent([&](Event event) {
        if (event == Event::Return) {
            if (selected == 0) {
                std::cout << "カメラ取込みを実行します...\n";
                std::this_thread::sleep_for(std::chrono::seconds(1));
            } else if (selected == 1) {
                std::cout << "画像処理テストを実行します...\n";
                std::this_thread::sleep_for(std::chrono::seconds(1));
            } else if (selected == 2) {
                std::cout << "送信テストを実行します...\n";
                std::this_thread::sleep_for(std::chrono::seconds(1));
            } else if (selected == 3) {
                screen.Exit();
            }
            return true; // イベント処理済み
        }
        return false;
    });

    screen.Loop(menu);
    return 0;
}


#include <opencv2/opencv.hpp>
#include <cmath>

void sobel_magnitude_32f_C1_borderDefault(const cv::Mat& src, cv::Mat& dst)
{
    CV_Assert(src.type() == CV_32FC1);

    dst.create(src.size(), CV_32FC1);
    if (src.rows < 1 || src.cols < 1) { dst.setTo(0); return; }
    if (src.rows == 1 || src.cols == 1) { dst.setTo(0); return; } // 3x3未満は勾配ゼロ扱い

    // 1pxパディング（BORDER_DEFAULT == BORDER_REFLECT_101）
    cv::Mat pad;
    cv::copyMakeBorder(src, pad, 1, 1, 1, 1, cv::BORDER_DEFAULT);

    const int rows = src.rows;
    const int cols = src.cols;

    // pad は (rows+2) x (cols+2) なので、中心(1..rows, 1..cols)が元画像に対応
    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& r){
        for (int y = r.start; y < r.end; ++y)
        {
            const float* p0 = pad.ptr<float>(y + 0); // (y-1) in src
            const float* p1 = pad.ptr<float>(y + 1); // (y)
            const float* p2 = pad.ptr<float>(y + 2); // (y+1)
            float*       pd = dst.ptr<float>(y);

            // x は pad では 1..cols が元画像の 0..cols-1 に対応
            for (int x = 0; x < cols; ++x)
            {
                const int xx = x + 1;

                const float gx =
                      (p0[xx + 1] - p0[xx - 1])
                    + 2.0f * (p1[xx + 1] - p1[xx - 1])
                    + (p2[xx + 1] - p2[xx - 1]);

                const float gy =
                      (p2[xx - 1] + 2.0f * p2[xx] + p2[xx + 1])
                    - (p0[xx - 1] + 2.0f * p0[xx] + p0[xx + 1]);

                pd[x] = std::sqrt(gx * gx + gy * gy);   // L2ノルム
                // pd[x] = std::fabs(gx) + std::fabs(gy); // L1ノルムにしたい場合
            }
        }
    });
}


cv::Mat mag(src.size(), CV_32F);

cv::parallel_for_(cv::Range(1, src.rows-1), [&](const cv::Range& r){
    for (int y = r.start; y < r.end; ++y) {
        const uchar* p0 = src.ptr<uchar>(y-1);
        const uchar* p1 = src.ptr<uchar>(y);
        const uchar* p2 = src.ptr<uchar>(y+1);
        float* pm = mag.ptr<float>(y);
        for (int x = 1; x < src.cols-1; ++x) {
            int gx =  (p0[x+1] - p0[x-1])
                    + 2*(p1[x+1] - p1[x-1])
                    +   (p2[x+1] - p2[x-1]);
            int gy =  (p2[x-1] + 2*p2[x] + p2[x+1])
                    - (p0[x-1] + 2*p0[x] + p0[x+1]);
            pm[x] = std::sqrt(float(gx*gx + gy*gy));   // or |gx|+|gy|
        }
    }
});


using System;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;

internal static class Itt
{
    private const string Dll = "ittnotify_collector.dll";

    // ---- ITT native (cdecl + ANSI が大事) ----
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern IntPtr __itt_domain_create(string name);

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern IntPtr __itt_string_handle_create(string name);

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
    private static extern void __itt_task_begin(
        IntPtr domain, IntPtr parent, IntPtr id, IntPtr handle);

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
    private static extern void __itt_task_end(IntPtr domain);

    // optional: 瞬間マーカー（縦線）
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    private static extern void __itt_marker(
        IntPtr domain, int scope /*__itt_marker_scope*/, string name, IntPtr id);

    private static readonly IntPtr Domain;
    private static readonly bool Available;
    private static readonly ConcurrentDictionary<string, IntPtr> HandleCache = new();

    static Itt()
    {
        try
        {
            Domain = __itt_domain_create("CSharp");
            Available = Domain != IntPtr.Zero;
        }
        catch
        {
            Available = false; // DLL未配置でもアプリは落とさない
        }
    }

    private static IntPtr GetHandle(string name)
        => HandleCache.GetOrAdd(name, n => __itt_string_handle_create(n));

    /// <summary>区間開始（タイムラインに「帯」）</summary>
    public static void Begin(string taskName)
    {
        if (!Available) return;
        try { __itt_task_begin(Domain, IntPtr.Zero, IntPtr.Zero, GetHandle(taskName)); }
        catch { /* ignore */ }
    }

    /// <summary>区間終了</summary>
    public static void End()
    {
        if (!Available) return;
        try { __itt_task_end(Domain); } catch { /* ignore */ }
    }

    /// <summary>using で安全に区間マーカー</summary>
    public sealed class Scope : IDisposable
    {
        private readonly bool _begun;
        public Scope(string taskName)
        {
            if (Available)
            {
                try
                {
                    __itt_task_begin(Domain, IntPtr.Zero, IntPtr.Zero, GetHandle(taskName));
                    _begun = true;
                }
                catch { _begun = false; }
            }
        }
        public void Dispose()
        {
            if (_begun)
            {
                try { __itt_task_end(Domain); } catch { /* ignore */ }
            }
        }
    }

    /// <summary>瞬間マーカー（縦線）。scope=2 は task スコープ相当</summary>
    public static void Mark(string name, int scope = 2)
    {
        if (!Available) return;
        try { __itt_marker(Domain, scope, name, IntPtr.Zero); } catch { /* ignore */ }
    }
}


using System;
using System.Runtime.InteropServices;

internal static class IttApi
{
    private const string Dll = "ittnotify.dll";

    [DllImport(Dll)]
    private static extern IntPtr __itt_domain_create(string name);

    [DllImport(Dll)]
    private static extern IntPtr __itt_string_handle_create(string name);

    [DllImport(Dll)]
    private static extern void __itt_task_begin(IntPtr domain, IntPtr parent, IntPtr id, IntPtr handle);

    [DllImport(Dll)]
    private static extern void __itt_task_end(IntPtr domain);

    // 使いやすいラッパ
    private static readonly IntPtr domain = __itt_domain_create("CSharpTasks");

    public static void Begin(string name)
    {
        var handle = __itt_string_handle_create(name);
        __itt_task_begin(domain, IntPtr.Zero, IntPtr.Zero, handle);
    }

    public static void End()
    {
        __itt_task_end(domain);
    }
}

using System;
using System.Threading;
using System.Threading.Tasks;

class Program
{
    static async Task Main()
    {
        Console.WriteLine("Start");

        var tasks = new[]
        {
            Task.Run(() => Work("TaskA")),
            Task.Run(() => Work("TaskB")),
            Task.Run(() => Work("TaskC")),
        };

        await Task.WhenAll(tasks);
    }

    static void Work(string name)
    {
        IttApi.Begin(name);      // ← VTune上で帯が始まる
        Thread.Sleep(3000);      // 擬似処理
        IttApi.End();            // ← 帯が終わる
    }
}

using System;
using System.Threading;
using System.Threading.Tasks;

class Program
{
    static async Task Main()
    {
        var tasks = new[]
        {
            Task.Run(() => Work("Thread-A")),
            Task.Run(() => Work("Thread-B")),
            Task.Run(() => Work("Thread-C")),
        };

        await Task.WhenAll(tasks);
    }

    static void Work(string threadName)
    {
        // ここでスレッド名を設定
        Thread.CurrentThread.Name = threadName;

        Console.WriteLine($"[{threadName}] start");
        Thread.Sleep(3000);
        Console.WriteLine($"[{threadName}] end");
    }
}








using System;
using System.Diagnostics;

class Program
{
    static void Main()
    {
        const int N = 1 << 28; // 約2億要素 ≈ 1.6GB (double×3配列)
        double[] A = new double[N];
        double[] B = new double[N];
        double[] C = new double[N];
        double alpha = 3.14;

        // 初期化
        for (int i = 0; i < N; i++)
        {
            A[i] = 1.0;
            B[i] = 2.0;
            C[i] = 3.0;
        }

        // ウォームアップ（JITの影響を除く）
        for (int w = 0; w < 2; w++)
        {
            for (int i = 0; i < N; i++)
                A[i] = B[i] + alpha * C[i];
        }

        // 計測
        Stopwatch sw = Stopwatch.StartNew();
        for (int i = 0; i < N; i++)
            A[i] = B[i] + alpha * C[i];
        sw.Stop();

        double seconds = sw.Elapsed.TotalSeconds;

        // 理論的なメモリアクセス量：B,Cから読み込み16B + Aへ書き込み8B = 24B
        double bytesMoved = N * 24.0;
        double gbps = bytesMoved / (seconds * 1e9);

        Console.WriteLine($"Time = {seconds:F3} s, Bandwidth ≈ {gbps:F1} GB/s");
    }
}



// Simple STREAM-like triad (C++17, x64 Release, /O2, /arch:AVX2/AVX512)
#include <chrono>
#include <vector>
#include <cstdio>
int main(){
    const size_t N = (size_t)1<<28; // ~2.1e8 elements ≈ 800MB for doubles
    std::vector<double> A(N,1.0), B(N,2.0), C(N,3.0);
    const double alpha = 3.14;
    // warmup
    for(int w=0; w<2; ++w) for(size_t i=0;i<N;++i) A[i]=B[i]+alpha*C[i];

    auto t0 = std::chrono::high_resolution_clock::now();
    for(size_t i=0;i<N;++i) A[i]=B[i]+alpha*C[i]; // 2 loads + 1 load + 1 store
    auto t1 = std::chrono::high_resolution_clock::now();
    double s = std::chrono::duration<double>(t1-t0).count();
    // bytes moved ≈ (reads B,C) 16B + (write A) 8B = 24B per iter (理想化)
    double gbps = (N * 24.0) / (s * 1e9);
    std::printf("Time=%.3fs  Bandwidth≈%.1f GB/s\n", s, gbps);
}



using System;
using System.Globalization;
using System.Windows.Data;
using System.Windows.Media.Imaging;

public class PathToBitmapConverter : IValueConverter
{
    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        var path = value as string;
        if (string.IsNullOrEmpty(path)) return null;

        try
        {
            var bmp = new BitmapImage();
            bmp.BeginInit();
            bmp.CacheOption = BitmapCacheOption.OnLoad;  // ロック回避
            bmp.CreateOptions = BitmapCreateOptions.IgnoreImageCache;
            bmp.UriSource = new Uri(path, UriKind.Absolute);
            bmp.EndInit();
            bmp.Freeze();  // スレッドセーフ化
            return bmp;
        }
        catch
        {
            return null;
        }
    }

    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        return null; // 逆方向の変換は不要
    }
}

<Window.Resources>
    <local:PathToBitmapConverter x:Key="PathToBitmapConverter"/>
</Window.Resources>

xmlns:local="clr-namespace:YourAppNamespace"





1.	概要
	•	アプリの目的・機能概要
	•	対象読者（開発者向け）
	2.	環境構築・導入手順
	•	ソースコードのビルド
	•	インストール手順
	•	初期設定（DB接続やconfigファイルなど）
	3.	アプリの操作方法（利用者として使う部分）
	•	起動方法
	•	メニューや画面ごとの操作説明
	•	入力項目の説明（必須／任意、形式など）
	•	代表的なエラーと対処方法
	4.	運用・保守
	•	ログの場所と見方
	•	バックアップ／復旧手順
	•	よくある障害と対応フロー
	5.	開発・改修向け
	•	ソースコード構成
	•	外部ライブラリ／依存関係
	•	ビルド・リリース手順
	•	今後の改修時の注意点

第5章: 技術仕様・アルゴリズム解説
	•	5.1 入力画像の前処理
	•	5.2 特徴点抽出アルゴリズム
	•	5.3 フィルタリング／マッチング手法
	•	5.4 出力データの形式


1.	概要
	2.	環境構築・インストール手順
	3.	運用・操作方法
	4.	保守・トラブルシュート
	5.	技術仕様・アルゴリズム解説（開発者向け）
	6.	付録（APIリファレンス、サンプルデータなど）



// デバッグフォルダ作成

#include <windows.h>
#include <shlobj.h>   // SHGetFolderPath
#include <filesystem>
#include <iostream>

#include <string>
#include <windows.h>

std::string WStringToUtf8(const std::wstring& wstr)
{
    if (wstr.empty()) return std::string();
    int sizeNeeded = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(),
                                         (int)wstr.size(), NULL, 0, NULL, NULL);
    std::string str(sizeNeeded, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(),
                        (int)wstr.size(), &str[0], sizeNeeded, NULL, NULL);
    return str;
}

int main()
{
    std::wstring ws = L"C:\\Users\\星加\\AppData\\Local\\HoshikaWorks\\MyApp\\Debug";
    std::string utf8 = WStringToUtf8(ws);

    printf("UTF-8: %s\n", utf8.c_str());
}

int main() {
    wchar_t path[MAX_PATH];
    if (SUCCEEDED(SHGetFolderPathW(NULL, CSIDL_LOCAL_APPDATA, NULL, 0, path))) {
        std::filesystem::path debugPath = std::filesystem::path(path) / L"MyApp" / L"Debug";
        std::filesystem::create_directories(debugPath);

        std::wcout << L"Debug folder: " << debugPath << std::endl;
    }
}



// DetectorAPI.h
#pragma once
#include <cstdint>

#ifdef _WIN32
  #define DLL_EXPORT __declspec(dllexport)
#else
  #define DLL_EXPORT
#endif

extern "C" {

// Detector インスタンス作成 / 解放
DLL_EXPORT void* CreateDetector();
DLL_EXPORT void  DestroyDetector(void* handle);

// パラメータ設定
DLL_EXPORT void  SetMinWhite(void* handle, int firstWhite, int secondWhite);
DLL_EXPORT void  SetBlackRange(void* handle, int minBlack, int maxBlack);

// 1ブロック追加して判定実行
// data: 8bit グレイ, width×height, stride バイト
// 結果: 1=found, 0=not found
DLL_EXPORT int   PushBlock(void* handle,
                           const uint8_t* data,
                           int width,
                           int height,
                           int stride,
                           int64_t* blackStart,
                           int64_t* blackEnd);

}

// DetectorAPI.cpp
#include "DetectorAPI.h"
#include "PatternDetectorCV_IntensityStrict.hpp"  // ←これまで作った検出器クラス

struct DetectorWrapper {
    PatternDetectorCV_IntensityStrict det;
};

extern "C" {

DLL_EXPORT void* CreateDetector() {
    return new DetectorWrapper();
}

DLL_EXPORT void DestroyDetector(void* handle) {
    delete static_cast<DetectorWrapper*>(handle);
}

DLL_EXPORT void SetMinWhite(void* handle, int firstWhite, int secondWhite) {
    auto* w = static_cast<DetectorWrapper*>(handle);
    w->det.minFirstWhite  = firstWhite;
    w->det.minSecondWhite = secondWhite;
}

DLL_EXPORT void SetBlackRange(void* handle, int minBlack, int maxBlack) {
    auto* w = static_cast<DetectorWrapper*>(handle);
    w->det.minBlack = minBlack;
    w->det.maxBlack = maxBlack;
}

DLL_EXPORT int PushBlock(void* handle,
                         const uint8_t* data,
                         int width,
                         int height,
                         int stride,
                         int64_t* blackStart,
                         int64_t* blackEnd)
{
    auto* w = static_cast<DetectorWrapper*>(handle);
    auto res = w->det.pushBlock(data, width, height, stride);
    if (res.found) {
        if (blackStart) *blackStart = res.blackStart;
        if (blackEnd)   *blackEnd   = res.blackEnd;
        return 1;
    }
    return 0;
}

} // extern "C"




int fourH = rF.rows;
int i = 0;

while (i < fourH) {
    int i0 = i;  // この試行の開始位置

    // 1) 白ラン（>= minFirstWhite）
    int w1 = 0;
    while (i < fourH && rF.at<float>(i) < whiteMaxBlackRatio) { ++w1; ++i; }
    if (w1 < minFirstWhite) {
        i = i0 + 1;           // ← ここが重要：開始位置を1行だけ進めて再試行
        continue;
    }

    // 2) 黒ラン（連続、長さ [minBlack, maxBlack]）
    int b = 0;
    int bStartIdx = i;        // 黒開始のインデックス
    while (i < fourH && rF.at<float>(i) >= blackMinBlackRatio) { ++b; ++i; }

    if (b >= minBlack && b <= maxBlack) {
        // 3) 後続の白ラン（>= minSecondWhite）
        int w2 = 0;
        int j = i;
        while (j < fourH && rF.at<float>(j) < whiteMaxBlackRatio) { ++w2; ++j; }

        if (w2 >= minSecondWhite) {
            // ヒット！
            out.found = true;
            out.blackStart = globalBaseLine_ - (3LL * blockH_) + bStartIdx;
            out.blackEnd   = out.blackStart + (b - 1);
            resetWindowAfterHit();
            globalBaseLine_ += blockH_;
            slideWindow();
            return out;
        }
    }

    // どこかの条件で失敗 → 開始位置を1行だけ進めて再試行
    i = i0 + 1;
}

// Sliding4BlocksStrict.hpp
#pragma once
#include <opencv2/opencv.hpp>
#include <deque>
#include <cstdint>

class Sliding4BlocksStrict {
public:
    // 期待パターン: 白(>=minFirstWhite) → 黒(連続[minBlack,maxBlack]) → 白(>=minSecondWhite)
    int   minFirstWhite  = 100;
    int   minBlack       = 200;   // 連続黒 下限
    int   maxBlack       = 300;   // 連続黒 上限
    int   minSecondWhite = 100;

    // 二値化：1=Otsu（推奨：4ブロックまとめて）、0=固定しきい（binThresh 使用）
    int   binarizeMode = 1;
    int   binThresh    = 120;     // 照明が安定なら固定の方が再現性UP

    // 縦方向ぼかし（白スジつぶし）。0/1=無効、奇数で推奨 31/51/61 など
    int   verticalBlurK = 51;

    // 黒/白のプロファイル判定（黒率 r ∈[0,1]）
    float whiteMaxBlackRatio = 0.20f; // 白判定： r < 0.20
    float blackMinBlackRatio = 0.60f; // 黒判定： r >= 0.60

    // 1Dプロファイルの移動平均窓（縦方向）。0/1=無効、奇数で 31/51/… 推奨
    int   profileSmoothW = 31;

    // 列方向 ROI（全幅: x=0, w=-1）
    int   roiX = 0, roiW = -1;

    struct Result {
        bool    found=false;
        int64_t blackStart=-1;  // グローバル行（inclusive）
        int64_t blackEnd=-1;    // グローバル行（inclusive）
        int64_t blackCenter() const { return (found? (blackStart+blackEnd)/2 : -1); }
    };

    Sliding4BlocksStrict():blockW_(0),blockH_(0),globalBaseLine_(0){}

    // 生ポインタのブロックを追加。幅W/高さH/stride（連続ならW）
    // ブロックサイズは一定である前提（ラインセンサの典型仕様）
    Result pushBlock(const uint8_t* data, int W, int H, ptrdiff_t stride=-1){
        Result out;
        if (!data || W<=0 || H<=0) return out;
        if (stride < 0) stride = W;

        // 受け取ったポインタに Mat ヘッダを被せる（コピー無し）
        cv::Mat m(H, W, CV_8UC1, const_cast<uint8_t*>(data), (size_t)stride);

        // 初回のサイズ決定
        if (blocks_.empty()) { blockW_ = W; blockH_ = H; }

        // ROI を安全にクリップ
        int x0 = (roiX<0)?0:(roiX>W?W:roiX);
        int ww = (roiW>0? roiW : W);
        if (ww > W - x0) ww = W - x0;
        if (ww <= 0) { globalBaseLine_ += H; return out; }

        // ROI ビューを保存（コピー無し）。ただし元ポインタの寿命に注意。
        blocks_.push_back(m(cv::Rect(x0, 0, ww, H)).clone()); 
        // ↑ 上流バッファの寿命が短い場合も安全にするため、ここは clone() で確保。
        //   もし寿命保証されるなら clone を外しコピー無しでOK（速度↑メモリ↓）

        // 4個溜まるまで待つ
        if (blocks_.size() < 4) { globalBaseLine_ += H; return out; }

        // 4個を縦に結合（コピー）
        cv::Mat big;
        cv::vconcat(std::vector<cv::Mat>{blocks_[0],blocks_[1],blocks_[2],blocks_[3]}, big);

        // --- 二値化（黒=255, 白=0） ---
        cv::Mat bin;
        if (binarizeMode==1) {
            cv::threshold(big, bin, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
        } else {
            cv::threshold(big, bin, binThresh, 255, cv::THRESH_BINARY_INV);
        }

        // --- 縦方向だけ強ぼかし（任意） ---
        if (verticalBlurK >= 3 && (verticalBlurK%2)==1) {
            cv::boxFilter(bin, bin, -1, cv::Size(1, verticalBlurK), cv::Point(-1,-1), true, cv::BORDER_REPLICATE);
        }

        // --- 行ごとの黒率プロファイル r(y) ∈ [0,1] を作る ---
        cv::Mat rF; // (4H)×1, float
        cv::reduce(bin, rF, 1, cv::REDUCE_AVG, CV_32F);
        rF /= 255.0f;

        // --- 1D 平滑（任意） ---
        if (profileSmoothW >= 3 && (profileSmoothW%2)==1) {
            cv::blur(rF, rF, cv::Size(1, profileSmoothW));
        }

        // --- 厳格な連続 [minBlack, maxBlack] 検索（前後の白も条件に含める） ---
        // まず白区間 >= minFirstWhite を探す
        int fourH = rF.rows;
        int i = 0;
        while (i < fourH) {
            // 白ラン
            int w1 = 0;
            while (i < fourH && rF.at<float>(i) < whiteMaxBlackRatio) { ++w1; ++i; }
            if (w1 < minFirstWhite) { ++i; continue; } // 条件未達→次へ

            // 黒ラン（連続のみ）
            int b = 0; int bStartIdx = i;
            while (i < fourH && rF.at<float>(i) >= blackMinBlackRatio) { ++b; ++i; }
            if (b >= minBlack && b <= maxBlack) {
                // 後続の白ラン
                int w2 = 0; int j = i;
                while (j < fourH && rF.at<float>(j) < whiteMaxBlackRatio) { ++w2; ++j; }
                if (w2 >= minSecondWhite) {
                    out.found = true;
                    out.blackStart = globalBaseLine_ - (3LL * blockH_) + bStartIdx; // 窓先頭のグローバル行 = base - 3H
                    out.blackEnd   = out.blackStart + (b - 1);
                    // 検出したら状態を初期化（連続検出したいなら適宜変更）
                    resetWindowAfterHit();
                    // 4ブロック“使い切った”わけではないので、次回の base を正しく進める：
                    globalBaseLine_ += H; // 今回分の新規ブロックぶん進める
                    // スライド（1ブロック捨てて次へ）
                    slideWindow();
                    return out;
                }
            }
            // 条件を満たさなかった場合、次の候補へ
        }

        // 検出できなかった：1ブロック分スライド
        globalBaseLine_ += H;
        slideWindow();
        return out;
    }

    void setBlockSizeHint(int W, int H){ blockW_=W; blockH_=H; }

private:
    // 先頭ブロックを捨てる
    void slideWindow(){
        if (!blocks_.empty()) blocks_.pop_front();
    }
    void resetWindowAfterHit(){
        blocks_.clear();
    }

    std::deque<cv::Mat> blocks_; // ROI後の各ブロック（高さ=BH, 幅=ROI幅）
    int blockW_, blockH_;
    // 直近 pushBlock 呼び出し時点を最下段としたとき、
    // 4ブロック窓の先頭（最上段）のグローバル行は (globalBaseLine_ - 3*BH)
    int64_t globalBaseLine_;
};

#include "Sliding4BlocksStrict.hpp"
#include <vector>
#include <iostream>

int main(){
    Sliding4BlocksStrict det;
    det.minFirstWhite  = 100;
    det.minBlack       = 200;
    det.maxBlack       = 300;
    det.minSecondWhite = 100;
    det.binarizeMode   = 1;   // Otsu（4ブロックまとめた窓）
    det.verticalBlurK  = 51;  // 縦強ぼかし（白スジ潰し）
    det.profileSmoothW = 31;  // プロファイルも少し平滑
    // det.roiX = 100; det.roiW = 800; // 必要なら列方向ROI

    // ダミー：各ブロック高さBH=128、幅W=1024 とする
    const int W=1024, BH=128;
    auto makeBlock = [&](int startBlackY, int endBlackY, int blockIndex){
        std::vector<uint8_t> v(W*BH, 235); // 白
        int base = blockIndex*BH;
        for(int y=0;y<BH;y++){
            int gy = base + y;
            if (gy>=startBlackY && gy<=endBlackY)
                std::fill_n(v.data()+y*W, W, 20); // 黒
        }
        return v;
    };

    // 4ブロックめで黒帯が完成する（例：白100→黒240→白100）
    std::vector<std::vector<uint8_t>> blocks;
    for(int k=0;k<6;k++){
        blocks.push_back(makeBlock(100, 339, k)); // 240行黒（100..339）
    }

    for (int k=0;k<(int)blocks.size();k++){
        auto res = det.pushBlock(blocks[k].data(), W, BH, W);
        if (res.found) {
            std::cout << "FOUND: start=" << res.blackStart
                      << " end=" << res.blackEnd
                      << " center=" << res.blackCenter() << "\n";
        }
    }
}




// Otsu 二値（黒→255）※反転しないなら、後の判定ロジックを合わせてね
cv::Mat bin;
cv::threshold(view, bin, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

// 1) 白スジ埋め（縦方向 Close）
cv::Mat kClose = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1,5));
cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, kClose, cv::Point(-1,-1), 1);

// 2) 黒の塩粒除去（横方向 Open）
cv::Mat kOpen  = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,1));
cv::morphologyEx(bin, bin, cv::MORPH_OPEN,  kOpen,  cv::Point(-1,-1), 1);

// 行ごとの黒率を算出して、連続 [minBlack,maxBlack] を判定
cv::Mat blackRatioF;
cv::reduce(bin, blackRatioF, 1, cv::REDUCE_AVG, CV_32F);
blackRatioF /= 255.0f;

// PatternDetectorCV_IntensityStrict.hpp
#pragma once
#include <opencv2/opencv.hpp>
#include <cstdint>

class PatternDetectorCV_IntensityStrict {
public:
    // 期待パターン: 白(>=minFirstWhite) -> 黒(連続[minBlack,maxBlack]) -> 白(>=minSecondWhite)
    int   minFirstWhite  = 100;
    int   minBlack       = 200;
    int   maxBlack       = 300;
    int   minSecondWhite = 100;

    // ヒステリシスしきい（平均輝度 0..255）
    // 例: 白判定は >= 180、黒判定は <= 120（間は前状態を維持）
    int   thrWhiteHigh = 180;  // 白確定境界（高いほど白に厳しい）
    int   thrBlackLow  = 120;  // 黒確定境界（低いほど黒に厳しい）

    // 行平均を少し平滑化（0/1=無効, >=2 で指数移動平均）
    int   emaStrength = 0;

    // 列方向 ROI（全幅: x=0, w=-1）
    int   roiX = 0, roiW = -1;

    struct Result {
        bool    found = false;
        int64_t blackStart = -1; // inclusive (global line)
        int64_t blackEnd   = -1; // inclusive
        int64_t blackCenter() const { return (blackStart>=0 && blackEnd>=0) ? ((blackStart+blackEnd)/2) : -1; }
    };

    PatternDetectorCV_IntensityStrict(){ reset(); }
    void reset(){
        state_ = State::FirstWhite; cnt_ = 0; globalLine_ = 0;
        blackStart_ = -1; blackEnd_ = -1; emaValid_ = false; ema_ = 0.f;
        lastStateBW_ = 0; // 1=white, -1=black, 0=unknown
    }

    // 生ポインタのブロック投入（8bitグレイ）
    Result pushBlock(const uint8_t* data, int width, int height, ptrdiff_t stride=-1){
        Result out;
        if (!data || width<=0 || height<=0) { globalLine_ += (height>0?height:0); return out; }
        if (stride < 0) stride = width;

        cv::Mat block(height, width, CV_8UC1, const_cast<uint8_t*>(data), (size_t)stride);

        // ROI クリップ
        int x0 = (roiX < 0) ? 0 : (roiX > width ? width : roiX);
        int ww = (roiW > 0) ? roiW : width;
        if (ww > width - x0) ww = width - x0;
        if (ww <= 0) { globalLine_ += height; return out; }

        cv::Mat view = block(cv::Rect(x0, 0, ww, height));

        // 行平均(0..255)を一括計算
        cv::Mat rowMeanF;
        cv::reduce(view, rowMeanF, 1, cv::REDUCE_AVG, CV_32F);

        for (int i=0; i<rowMeanF.rows; ++i){
            float m = rowMeanF.at<float>(i);

            // EMA 平滑（任意）
            if (emaStrength >= 2){
                float alpha = 1.0f / (float)emaStrength;
                if (!emaValid_) { ema_ = m; emaValid_ = true; }
                else            { ema_ = (1.0f - alpha)*ema_ + alpha*m; }
            } else {
                ema_ = m;
                emaValid_ = true;
            }

            // ヒステリシス判定
            int bw = lastStateBW_;
            if (ema_ >= thrWhiteHigh) bw = 1;      // 白確定
            else if (ema_ <= thrBlackLow) bw = -1; // 黒確定
            // 中間帯では bw を変えない（直前状態を維持）

            switch (state_) {
            case State::FirstWhite:
                if (bw == 1) { if (++cnt_ >= minFirstWhite) { state_ = State::Black; cnt_ = 0; } }
                else         { cnt_ = 0; }
                break;

            case State::Black:
                if (bw == -1) {
                    if (cnt_ == 0) blackStart_ = globalLine_ + i;
                    ++cnt_;
                    if (maxBlack > 0 && cnt_ > maxBlack) {
                        // 上限オーバー：この行から新しい連続黒として再スタート
                        blackStart_ = globalLine_ + i;
                        cnt_ = 1;
                    }
                } else {
                    // 連続黒が途切れた。長さチェック。
                    if (cnt_ >= minBlack && (maxBlack==0 || cnt_ <= maxBlack)) {
                        blackEnd_ = globalLine_ + i - 1;
                        state_ = State::SecondWhite; cnt_ = 0;
                    } else {
                        // 失敗：白からやり直し
                        state_ = State::FirstWhite;
                        cnt_ = (bw == 1) ? 1 : 0;
                        blackStart_ = -1;
                    }
                }
                break;

            case State::SecondWhite:
                if (bw == 1) { if (++cnt_ >= minSecondWhite) {
                        out.found = true;
                        out.blackStart = blackStart_;
                        out.blackEnd   = blackEnd_;
                        reset();
                        globalLine_ += (rowMeanF.rows - (i+1));
                        return out;
                    } }
                else { cnt_ = 0; }
                break;
            }

            lastStateBW_ = bw;
        }

        globalLine_ += rowMeanF.rows;
        return out;
    }

private:
    enum class State { FirstWhite, Black, SecondWhite } state_;
    int64_t globalLine_ = 0;
    int     cnt_ = 0;
    int64_t blackStart_ = -1, blackEnd_ = -1;

    // ヒステリシスのための直前状態
    int     lastStateBW_ = 0; // 1=white, -1=black, 0=unknown
    // EMA
    bool    emaValid_ = false;
    float   ema_ = 0.f;
};



// PatternDetectorCV_Fuzzy.hpp
#pragma once
#include <opencv2/opencv.hpp>
#include <cstdint>
#include <cmath>
#include <limits>

class PatternDetectorCV_Fuzzy {
public:
    // --- 期待パターン: (白 >= minFirstWhite) -> (黒 >= minBlack) -> (白 >= minSecondWhite) ---
    int minFirstWhite  = 100;
    int minBlack       = 180;  // ← 200ピッタリでなく最小値でOKにする
    int minSecondWhite = 100;

    // 黒長の上限を付けたい場合（0なら無視）：例 240 行
    int maxBlack       = 0;    // 0 で無効（上限なし）

    // ヒステリシス：行の「黒率」(0..1) で判定
    //   白: black_ratio < whiteMaxBlackRatio
    //   黒: black_ratio >= blackMinBlackRatio
    // その中間（グレーゾーン）は“どちらでもない”
    float whiteMaxBlackRatio = 0.20f; // 白は黒率 < 20%
    float blackMinBlackRatio = 0.60f; // 黒は黒率 >= 60%

    // ギャップ許容（“その状態に相反する行”を何行まで連続で許すか）
    int  allowedWhiteGaps = 6; // 白カウント中に黒/灰が混ざってよい最大連続行
    int  allowedBlackGaps = 8; // 黒カウント中に白/灰が混ざってよい最大連続行

    // 二値化モード：1=Otsu（推奨）、0=固定しきい（binThresh使用）
    int  binarizeMode = 1;
    int  binThresh    = 128;

    // 列方向ROI（全幅: x=0, w=-1）
    int roiX = 0, roiW = -1;

    struct Result {
        bool    found = false;
        int64_t blackStart = -1; // 黒開始（inclusive, global）
        int64_t blackEnd   = -1; // 黒終了（inclusive, global）
        int64_t blackCenter() const { return (blackStart>=0 && blackEnd>=0)? (blackStart+blackEnd)/2 : -1; }
    };

    PatternDetectorCV_Fuzzy(){ reset(); }
    void reset(){
        state_ = State::FirstWhite;
        cnt_ = 0; gap_ = 0;
        globalLine_ = 0;
        blackStart_ = -1; blackEnd_ = -1; lastBlackLine_ = -1;
    }

    // 生ポインタのブロックを投入（8bitグレイ）
    Result pushBlock(const uint8_t* data, int width, int height, ptrdiff_t stride = -1) {
        Result out;
        if (!data || width<=0 || height<=0) { globalLine_ += (height>0?height:0); return out; }
        if (stride < 0) stride = width;

        // ポインタにヘッダを被せる（ノーコピー）
        cv::Mat block(height, width, CV_8UC1, const_cast<uint8_t*>(data), (size_t)stride);

        // ROI クリップ
        int x0 = roiX < 0 ? 0 : (roiX > width ? width : roiX);
        int ww = (roiW>0 ? roiW : width);
        if (ww > width - x0) ww = width - x0;
        if (ww <= 0) { globalLine_ += height; return out; }

        cv::Mat view = block(cv::Rect(x0, 0, ww, height));

        // 二値化（黒=255, 白=0）
        cv::Mat bin;
        if (binarizeMode==1) {
            cv::threshold(view, bin, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
        } else {
            cv::threshold(view, bin, binThresh, 255, cv::THRESH_BINARY_INV);
        }

        // 行ごとの黒率 (0..1)
        cv::Mat blackRatioF;
        cv::reduce(bin, blackRatioF, 1, cv::REDUCE_AVG, CV_32F);
        blackRatioF /= 255.0f;

        // ラインごとに状態機械を進める
        for (int i=0; i<blackRatioF.rows; ++i) {
            float r = blackRatioF.at<float>(i);
            bool isWhite = (r < whiteMaxBlackRatio);
            bool isBlack = (r >= blackMinBlackRatio);

            switch (state_) {
            case State::FirstWhite:
                if (isWhite) { ++cnt_; gap_ = 0; }
                else {
                    if (++gap_ > allowedWhiteGaps) { cnt_ = 0; gap_ = 0; } // 連続ギャップ超えで白カウントやり直し
                }
                if (cnt_ >= minFirstWhite) { state_=State::Black; cnt_=0; gap_=0; }
                break;

            case State::Black:
                if (isBlack) {
                    if (cnt_ == 0) blackStart_ = globalLine_ + i;
                    ++cnt_; gap_ = 0;
                    lastBlackLine_ = globalLine_ + i; // 最新の“真に黒”行を覚える
                    if (maxBlack > 0 && cnt_ > maxBlack) { // 上限を超えたら失敗としてリセット（必要ないなら無視）
                        cnt_ = 0; gap_ = 0; blackStart_ = -1; lastBlackLine_ = -1;
                        state_ = State::FirstWhite;
                    }
                } else {
                    // 黒でない行（白/灰）：ギャップとして許容
                    if (++gap_ > allowedBlackGaps) {
                        // 許容超え → 黒カウントリセット
                        cnt_ = 0; gap_ = 0; blackStart_ = -1; lastBlackLine_ = -1;
                        state_ = State::FirstWhite; // 先頭白からやり直す（厳密運用）
                        break;
                    }
                }
                if (cnt_ >= minBlack) {
                    // 最小長さは満たした → 次の白へ遷移
                    blackEnd_ = (lastBlackLine_ >= 0) ? lastBlackLine_ : (globalLine_ + i);
                    state_ = State::SecondWhite; cnt_ = 0; gap_ = 0;
                }
                break;

            case State::SecondWhite:
                if (isWhite) { ++cnt_; gap_ = 0; }
                else {
                    if (++gap_ > allowedWhiteGaps) { cnt_ = 0; gap_ = 0; } // 許容超えで白やり直し
                }
                if (cnt_ >= minSecondWhite) {
                    out.found = true;
                    out.blackStart = blackStart_;
                    out.blackEnd   = blackEnd_;
                    reset();                          // 1パターン検出後は初期化（必要なら連続検出に変えてOK）
                    // このブロック以降の行番号調整
                    globalLine_ += (blackRatioF.rows - (i+1));
                    return out;
                }
                break;
            }
        }

        globalLine_ += blackRatioF.rows;
        return out;
    }

private:
    enum class State { FirstWhite, Black, SecondWhite } state_;
    int64_t globalLine_ = 0;
    int     cnt_ = 0;
    int     gap_ = 0;

    int64_t blackStart_ = -1;
    int64_t blackEnd_   = -1;
    int64_t lastBlackLine_ = -1; // 直近の“真に黒”行
};

#include "PatternDetectorCV_Fuzzy.hpp"
#include <vector>
#include <iostream>

int main(){
    PatternDetectorCV_Fuzzy det;

    // 許容を広めに
    det.minFirstWhite  = 90;     // 100の代わりに 90〜OK
    det.minBlack       = 180;    // 200の代わりに 180〜OK
    det.minSecondWhite = 90;
    det.maxBlack       = 240;    // 上限付けたいなら（不要なら 0 のまま）

    det.whiteMaxBlackRatio = 0.25f; // 白は黒率 < 25%
    det.blackMinBlackRatio = 0.55f; // 黒は黒率 >= 55%
    det.allowedWhiteGaps = 8;   // 白の途切れ許容
    det.allowedBlackGaps = 12;  // 黒の途切れ許容
    det.binarizeMode     = 1;   // Otsu

    // ダミー画像（H 行中、ややガタつく黒帯 190 行）
    const int W=2048, H=450;
    std::vector<uint8_t> buf(W*H, 230); // 白
    // 黒帯：110..299 に黒、合間にところどころ白を混ぜる
    for (int y=110; y<=299; ++y) {
        uint8_t v = (y%17==0) ? 200 : 30; // たまに明るめ(=穴)を混ぜる
        std::fill_n(buf.data()+y*W, W, v);
    }

    auto res = det.pushBlock(buf.data(), W, H, W);
    if (res.found) {
        std::cout << "FOUND: blackStart="<<res.blackStart
                  << " blackEnd="<<res.blackEnd
                  << " center="<<res.blackCenter() << "\n";
    } else {
        std::cout << "NOT FOUND\n";
    }
}



// PatternDetectorCV_Ratio.hpp
#pragma once
#include <opencv2/opencv.hpp>
#include <cstdint>
#include <cmath>

class PatternDetectorCV_Ratio {
public:
    int firstWhiteNeeded  = 100;
    int blackNeeded       = 200;
    int secondWhiteNeeded = 100;

    // 黒率で判定：白=黒率 < whiteMaxBlackRatio、黒=黒率 >= blackMinBlackRatio
    // 例: 白は <0.2、黒は >=0.6 とみなす（環境に合わせて調整）
    float whiteMaxBlackRatio = 0.2f;
    float blackMinBlackRatio = 0.6f;

    // 0: 固定二値 (binThresh>=0), 1: Otsu（推奨）
    int binarizeMode = 1;
    int binThresh = 128; // binarizeMode==0 のとき有効

    // 列方向ROI（全幅なら x=0, w=-1）
    int roiX = 0, roiW = -1;

    struct Result {
        bool   found = false;
        int64_t blackStart = -1;
        int64_t blackEnd   = -1;
        int64_t blackCenter() const { return (blackStart>=0 && blackEnd>=0)? (blackStart+blackEnd)/2 : -1; }
    };

    void reset(){
        state_ = State::FirstWhite; cnt_=0; globalLine_=0;
        blackStart_=-1; blackEnd_=-1;
    }
    PatternDetectorCV_Ratio(){ reset(); }

    // 生ポインタのブロックを投入（8bitグレイ）
    Result pushBlock(const uint8_t* data, int width, int height, ptrdiff_t stride=-1) {
        Result out;
        if (!data || width<=0 || height<=0) { globalLine_ += (height>0?height:0); return out; }
        if (stride < 0) stride = width;

        // ノーコピーでMat化
        cv::Mat block(height, width, CV_8UC1, const_cast<uint8_t*>(data), (size_t)stride);

        // ROIクリップ（clamp使わない）
        int x0 = roiX < 0 ? 0 : (roiX > width ? width : roiX);
        int ww = (roiW>0 ? roiW : width);
        if (ww > width - x0) ww = width - x0;
        if (ww <= 0) { globalLine_ += height; return out; }

        cv::Mat view = block(cv::Rect(x0, 0, ww, height));

        // 二値化（黒=255, 白=0）
        cv::Mat bin;
        if (binarizeMode==1) {
            cv::threshold(view, bin, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
        } else {
            cv::threshold(view, bin, binThresh, 255, cv::THRESH_BINARY_INV);
        }

        // 行ごとの黒率（0..1）
        cv::Mat blackCountF;
        cv::reduce(bin, blackCountF, 1, cv::REDUCE_AVG, CV_32F); // 0..255 の平均
        // 255で割る→0..1
        blackCountF /= 255.0f;

        // 状態機械
        for (int i=0;i<blackCountF.rows;i++){
            float r = blackCountF.at<float>(i);
            bool isWhite = (r < whiteMaxBlackRatio);
            bool isBlack = (r >= blackMinBlackRatio);

            switch(state_){
            case State::FirstWhite:
                if (isWhite) { if (++cnt_ >= firstWhiteNeeded) { state_=State::Black; cnt_=0; } }
                else cnt_=0; 
                break;

            case State::Black:
                if (isBlack) {
                    if (cnt_==0) blackStart_ = globalLine_ + i;
                    if (++cnt_ >= blackNeeded) { blackEnd_ = globalLine_ + i; state_=State::SecondWhite; cnt_=0; }
                } else { cnt_=0; blackStart_=-1; } // 途中で切れたら厳密にリセット
                break;

            case State::SecondWhite:
                if (isWhite) { if (++cnt_ >= secondWhiteNeeded) {
                        out.found=true; out.blackStart=blackStart_; out.blackEnd=blackEnd_;
                        reset(); globalLine_ += blackCountF.rows; return out;
                    } }
                else cnt_=0;
                break;
            }
        }

        globalLine_ += blackCountF.rows;
        return out;
    }

private:
    enum class State { FirstWhite, Black, SecondWhite } state_;
    int64_t globalLine_=0;
    int cnt_=0;
    int64_t blackStart_=-1, blackEnd_=-1;
};

#include "PatternDetectorCV_Ratio.hpp"
#include <vector>
#include <iostream>

int main() {
    PatternDetectorCV_Ratio det;
    det.firstWhiteNeeded  = 100;
    det.blackNeeded       = 200;
    det.secondWhiteNeeded = 100;
    det.whiteMaxBlackRatio = 0.2f; // 白は黒率 < 20%
    det.blackMinBlackRatio = 0.6f; // 黒は黒率 >= 60%
    det.binarizeMode = 1;          // Otsu

    // ダミー（W*H の白、途中200行だけ黒）
    const int W=2048, H=400;
    std::vector<uint8_t> buf(W*H, 230); // 多少グレーでもOK
    for(int y=100;y<300;y++) std::fill_n(buf.data()+y*W, W, 20);

    auto res = det.pushBlock(buf.data(), W, H, W);
    if (res.found) {
        std::cout << "FOUND: start="<<res.blackStart<<" end="<<res.blackEnd
                  << " center="<<res.blackCenter()<<"\n";
    } else {
        std::cout << "NOT FOUND\n";
    }
}

// PatternDetectorCV.hpp
#pragma once
#include <opencv2/opencv.hpp>
#include <cstdint>
#include <cmath>

class PatternDetectorCV {
public:
    // ---- パラメータ ----
    int firstWhiteNeeded  = 100;   // 最初の白 連続行数
    int blackNeeded       = 200;   // 黒 連続行数
    int secondWhiteNeeded = 100;   // 最後の白 連続行数

    // 白判定しきい値（0..255）。-1 なら自動（背景のランニング平均 + autoOffset）
    int   whiteThresh = 128;
    int   autoOffset  = 0;

    // 行平均の平滑（指数移動平均の係数）。0/1 で無効、>=2 で有効（大きいほどなめらか）
    int   emaStrength = 4;

    // 列方向 ROI（全幅なら x=0, w=-1）
    int roiX = 0;
    int roiW = -1;

    struct Result {
        bool   found = false;
        int64_t blackStart = -1;   // 黒の開始行（global, inclusive）
        int64_t blackEnd   = -1;   // 黒の終了行（global, inclusive）
        int64_t blackCenter() const { return (blackStart>=0 && blackEnd>=0)? ((blackStart+blackEnd)/2) : -1; }
    };

    PatternDetectorCV(){ reset(); }
    void reset(){
        state_ = State::FirstWhite;
        cnt_ = 0;
        globalLine_ = 0;
        blackStart_ = -1;
        blackEnd_   = -1;
        ema_ = std::numeric_limits<double>::quiet_NaN();
        bgMean_ = std::numeric_limits<double>::quiet_NaN();
    }

    // ---- ポインタで渡されたブロック（高さH行）を一括処理 ----
    // data: 先頭ポインタ（8bitグレイ）
    // width: 1行の画素数, height: 行数, stride: バイト単位（連続なら width）
    // 返り値: 最初に見つけたパターンのみ返す（複数検出したいならループを拡張）
    Result pushBlock(const uint8_t* data, int width, int height, ptrdiff_t stride = -1) {
        Result out;
        if (!data || width<=0 || height<=0) { globalLine_ += height>0?height:0; return out; }
        if (stride < 0) stride = width;

        // ポインタに Mat ヘッダを被せる（ノーコピー）
        cv::Mat block(height, width, CV_8UC1, const_cast<uint8_t*>(data), (size_t)stride);

        // ROI を安全にクリップ
        cv::Rect full(0, 0, width, height);
        cv::Rect roi(roiX, 0, (roiW>0? roiW : width), height);
        cv::Rect clipped = roi & full;
        if (clipped.width <= 0) { globalLine_ += height; return out; }

        // ROI 切り出し（ビュー：コピーなし）
        cv::Mat view = block(clipped);

        // 行平均を一括で計算（高さH × 幅Wroi → H×1 の float）
        cv::Mat rowMeanF;
        cv::reduce(view, rowMeanF, /*dim=*/1, cv::REDUCE_AVG, CV_32F); // 1次元方向に平均

        // 1行ずつ状態機械を進める
        for (int i = 0; i < rowMeanF.rows; ++i) {
            double mean = rowMeanF.at<float>(i);

            // EMA 平滑（任意）
            if (emaStrength >= 2) {
                double alpha = 1.0 / (double)emaStrength;
                if (std::isnan(ema_)) ema_ = mean;
                else                  ema_ = (1.0 - alpha) * ema_ + alpha * mean;
            } else {
                ema_ = mean;
            }

            // しきい値（固定 or 自動）
            int thr = whiteThresh;
            if (whiteThresh < 0) {
                // 背景のランニング平均（白っぽいときだけゆっくり更新）
                if (ema_ > 180.0) {
                    if (std::isnan(bgMean_)) bgMean_ = ema_;
                    else bgMean_ = 0.99 * bgMean_ + 0.01 * ema_;
                } else if (std::isnan(bgMean_)) {
                    bgMean_ = ema_;
                }
                thr = (int)std::round(std::clamp(bgMean_ + (double)autoOffset, 0.0, 255.0));
            }

            bool isWhite = (ema_ > thr);

            switch (state_) {
            case State::FirstWhite:
                if (isWhite) {
                    if (++cnt_ >= firstWhiteNeeded) {
                        state_ = State::Black;
                        cnt_ = 0;
                    }
                } else cnt_ = 0;
                break;

            case State::Black:
                if (!isWhite) {
                    if (cnt_ == 0) blackStart_ = globalLine_ + i;  // このブロック内のi行目
                    if (++cnt_ >= blackNeeded) {
                        blackEnd_ = globalLine_ + i;
                        state_ = State::SecondWhite;
                        cnt_ = 0;
                    }
                } else {
                    // 途中で黒が切れたらリセット（厳密運用）
                    cnt_ = 0;
                    blackStart_ = -1;
                }
                break;

            case State::SecondWhite:
                if (isWhite) {
                    if (++cnt_ >= secondWhiteNeeded) {
                        out.found = true;
                        out.blackStart = blackStart_;
                        out.blackEnd   = blackEnd_;
                        reset(); // 1パターン検出で初期化（継続検出にしたいならここを変更）
                        // ここで return すると、このブロック中の後続パターンは見ない
                        // 複数検出したい場合は out をリスト化し、ここでは続行に変える
                        // ただし今回の要件では1つ出ればOK想定
                        // ブロック末尾までの globalLine_ 加算を整合させるため、ここでは抜けずにフラグだけ立てるなら別処理が必要
                        // シンプルに抜けちゃいます：
                        globalLine_ += rowMeanF.rows; 
                        return out;
                    }
                } else cnt_ = 0;
                break;
            }
        }

        globalLine_ += rowMeanF.rows;
        return out;
    }

private:
    enum class State { FirstWhite, Black, SecondWhite } state_;

    int64_t globalLine_ = 0;
    int     cnt_ = 0;
    int64_t blackStart_ = -1;
    int64_t blackEnd_   = -1;

    double ema_   = std::numeric_limits<double>::quiet_NaN();
    double bgMean_= std::numeric_limits<double>::quiet_NaN();
};
#include "PatternDetectorCV.hpp"
#include <iostream>

int main() {
    PatternDetectorCV det;
    det.firstWhiteNeeded  = 100;
    det.blackNeeded       = 200;
    det.secondWhiteNeeded = 100;

    // 自動しきい値を使うなら
    // det.whiteThresh = -1;
    // det.autoOffset  = -10;
    // det.emaStrength = 6;

    // ROI を列方向で絞る場合
    // det.roiX = 100; det.roiW = 800;

    // ストリームから来たブロック（8bitグレイの生ポインタ）
    // 例: height=1024行ずつ、幅=2048、stride=2048（連続）
    // const uint8_t* buf = ... ;
    // auto res = det.pushBlock(buf, 2048, 1024, 2048);

    // ダミーで試すなら：
    const int W=512, H=400;
    std::vector<uint8_t> frame(W*H, 255);
    auto paint = [&](int y0, int y1, uint8_t v){
        for(int y=y0;y<=y1;y++){
            std::fill_n(frame.data()+y*W, W, v);
        }
    };
    paint(100, 299, 10); // 200行黒（100..299）

    auto res = det.pushBlock(frame.data(), W, H, W);
    if (res.found) {
        std::cout << "FOUND: blackStart=" << res.blackStart
                  << " blackEnd="   << res.blackEnd
                  << " center="     << res.blackCenter()
                  << std::endl;
    } else {
        std::cout << "not found\n";
    }
}



// PatternDetector.hpp
#pragma once
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>

class PatternDetector {
public:
    // --- パラメータ（必要に応じて調整） ---
    int firstWhiteNeeded  = 100;     // 先頭の白 連続行数
    int blackNeeded       = 200;     // 黒帯 連続行数
    int secondWhiteNeeded = 100;     // 後続の白 連続行数

    // 白/黒判定用
    // 平均輝度 > whiteThresh なら「白」。固定値 or 自動しきい（-1）を選択。
    int whiteThresh = 128;           // 0..255。-1 なら自動（ランニング平均+オフセット）
    int autoOffset  = 0;             // 自動時：基準に足す/引くオフセット（例：-10）

    // ノイズ抑制：移動平均（行の平均値に対して）
    // 0 なら無効、1 ならそのまま、2 以上で指数移動平均(EWMA)
    int emaStrength = 4;             // 推奨: 4〜8（大きいほどなめらか）

    // 列方向ROI（全幅を見るなら x0=0, w=-1）
    int roiX = 0;
    int roiW = -1;

    struct Result {
        bool   found = false;
        int64_t blackStart = -1;   // 黒の開始行 (inclusive)
        int64_t blackEnd   = -1;   // 黒の終了行 (inclusive)
        int64_t blackCenter() const { return (blackStart >= 0 && blackEnd >= 0) ? ((blackStart + blackEnd)/2) : -1; }
    };

    // 状態を初期化
    void reset() {
        state_ = State::FirstWhite;
        cnt_ = 0;
        globalLine_ = 0;
        blackStart_ = -1;
        blackEnd_ = -1;
        ema_ = NAN;
        bgMean_ = NAN;
    }

    PatternDetector() { reset(); }

    // 1行ずつ投入（8bitグレイ配列）。返り値で検出完了を通知。
    // row: 行先頭ポインタ / width: 全体の列数 / stride: バイト単位（連続なら width）
    Result pushLine(const uint8_t* row, int width, int stride = -1) {
        if (stride < 0) stride = width;
        Result out;

        // --- ROI 適用 ---
        int x0 = std::clamp(roiX, 0, width);
        int ww = (roiW > 0) ? std::min(roiW, width - x0) : (width - x0);
        if (ww <= 0) { ++globalLine_; return out; }

        // --- 行の平均輝度を計算（ROI 内） ---
        const uint8_t* p = row + x0;
        uint64_t sum = 0;
        for (int i = 0; i < ww; ++i) sum += p[i];
        double mean = static_cast<double>(sum) / ww;

        // --- EMA による平滑化（オプション）---
        if (emaStrength >= 2) {
            double alpha = 1.0 / static_cast<double>(emaStrength);
            if (std::isnan(ema_)) ema_ = mean;
            else                  ema_ = (1.0 - alpha) * ema_ + alpha * mean;
        } else {
            ema_ = mean;
        }

        // --- しきい値決定：固定 or 自動 ---
        int thr = whiteThresh;
        if (whiteThresh < 0) {
            // 自動：背景ランニング平均（白寄りのときに更新）＋オフセット
            if (isWhiteByFixed(ema_, 180)) {  // かなり白めのときに背景を更新（経験則）
                if (std::isnan(bgMean_)) bgMean_ = ema_;
                else bgMean_ = 0.99 * bgMean_ + 0.01 * ema_;
            } else if (std::isnan(bgMean_)) {
                bgMean_ = ema_; // 初期値
            }
            thr = static_cast<int>(std::clamp(bgMean_ + autoOffset, 0.0, 255.0));
        }

        bool isWhite = (ema_ > thr);

        // --- ステートマシン ---
        switch (state_) {
        case State::FirstWhite:
            if (isWhite) {
                if (++cnt_ >= firstWhiteNeeded) {
                    state_ = State::Black;
                    cnt_ = 0;
                }
            } else cnt_ = 0;
            break;

        case State::Black:
            if (!isWhite) {
                if (cnt_ == 0) blackStart_ = globalLine_; // 最初の黒行を記録
                cnt_++;
                if (cnt_ >= blackNeeded) {
                    blackEnd_ = globalLine_;              // 200本目の黒行
                    state_ = State::SecondWhite;
                    cnt_ = 0;
                }
            } else {
                // 黒が途切れた → リセット（厳密運用）
                cnt_ = 0;
                blackStart_ = -1;
            }
            break;

        case State::SecondWhite:
            if (isWhite) {
                if (++cnt_ >= secondWhiteNeeded) {
                    out.found = true;
                    out.blackStart = blackStart_;
                    out.blackEnd = blackEnd_;
                    reset(); // 1パターン検出でリセット（連続検出したいなら適宜変更）
                }
            } else cnt_ = 0;
            break;
        }

        ++globalLine_;
        return out;
    }

private:
    enum class State { FirstWhite, Black, SecondWhite } state_;

    // 内部状態
    int64_t globalLine_ = 0;
    int     cnt_ = 0;
    int64_t blackStart_ = -1;
    int64_t blackEnd_   = -1;

    // 平滑化・背景
    double ema_ = NAN;     // 行平均の指数移動平均
    double bgMean_ = NAN;  // 背景（白側）のランニング平均

    static bool isWhiteByFixed(double mean, int thr) { return mean > thr; }
};


#include "PatternDetector.hpp"
#include <vector>
#include <iostream>

int main() {
    PatternDetector det;

    // 必要なら調整
    det.firstWhiteNeeded  = 100;
    det.blackNeeded       = 200;
    det.secondWhiteNeeded = 100;

    // 自動しきい値を使うなら：
    // det.whiteThresh = -1;   // 自動
    // det.autoOffset  = -10;  // 背景より少し低めに設定して白判定を甘く
    // det.emaStrength = 6;    // 行平均を平滑化

    // 列方向の ROI を使うなら：
    // det.roiX = 100; det.roiW = 800;

    // ダミーデータ：ここでは幅 W の白/黒ラインを用意して順に投入する想定
    const int W = 2048;
    std::vector<uint8_t> white(W, 255);
    std::vector<uint8_t> black(W, 10);

    auto feed = [&](const std::vector<uint8_t>& line, int n){
        for (int i=0;i<n;i++) {
            auto res = det.pushLine(line.data(), W);
            if (res.found) {
                std::cout << "FOUND: blackStart=" << res.blackStart
                          << " blackEnd=" << res.blackEnd
                          << " blackCenter=" << res.blackCenter()
                          << std::endl;
            }
        }
    };

    feed(white, 100);
    feed(black, 200);
    feed(white, 100);
    return 0;
}





// 1行ずつ（もしくは小タイルずつ）投入して、マーカー開始yを返す。
// 返り値：開始y（0-basedのグローバル行番号）。未確定なら負値のまま。
struct MarkerStartDetector {
    // パラメータ
    int    required_on = 12;     // 連続ON必要行数（デバウンス）※環境に合わせ調整
    double black_ratio_th = 0.55;// 黒画素率の閾値（0〜1）
    int    bin_thresh = -1;      // 二値化閾値。-1なら大津自動（推奨：固定照明なら固定値でOK）
    int    avg_rows = 3;         // 行の移動平均（ノイズ低減）

    // 状態
    int64_t global_line = 0;     // これまでの累積行数
    int     on_streak = 0;       // 連続ONカウント
    int64_t start_y = -1;        // 確定した開始y（未確定は -1）
    std::deque<cv::Mat> row_buf; // 平均用の行バッファ（grayの1xW）

    // ROI（幅方向を限定したい場合）
    int roi_x = 0, roi_w = -1;   // -1 なら全幅

    void setROI(int x, int w){ roi_x = x; roi_w = w; }

    // 小タイル(複数行)でもOK。行ごとに処理。
    void pushRows(const cv::Mat& grayRows) {
        CV_Assert(grayRows.type()==CV_8UC1);
        for (int y = 0; y < grayRows.rows; ++y) {
            cv::Mat row = grayRows.row(y);

            // ROI 切り出し
            int W = row.cols;
            int x0 = std::max(0, roi_x);
            int ww = (roi_w>0 ? std::min(roi_w, W - x0) : W - x0);
            cv::Mat r = row.colRange(x0, x0 + ww);

            // 移動平均（行方向）：最新行をバッファして平均
            row_buf.push_back(r.clone());
            while ((int)row_buf.size() > avg_rows) row_buf.pop_back(); // ※単純保持（必要ならpop_frontで古い方を削除）
            // ↑dequeの積み方は用途により前後どちらでも良い。ここでは単純に直近avg_rows行だけ持つ想定。

            cv::Mat avg;
            if ((int)row_buf.size() == avg_rows) {
                cv::Mat acc = cv::Mat::zeros(1, r.cols, CV_32F);
                for (auto& rr : row_buf) {
                    cv::Mat f; rr.convertTo(f, CV_32F);
                    acc += f;
                }
                acc /= (float)avg_rows;
                acc.convertTo(avg, CV_8U);
            } else {
                avg = r; // たまるまで素通し
            }

            // 二値化（黒＝1）
            cv::Mat bin;
            if (bin_thresh < 0)
                cv::threshold(avg, bin, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
            else
                cv::threshold(avg, bin, bin_thresh, 255, cv::THRESH_BINARY_INV);

            // 黒画素率
            int black = cv::countNonZero(bin);
            double ratio = (double)black / (double)bin.cols;

            // 連続ON 判定（黒が多い行をON）
            bool is_on = (ratio >= black_ratio_th);
            if (is_on) {
                on_streak++;
                if (on_streak == required_on && start_y < 0) {
                    // ここで開始yを “最初のON” に戻したい場合は (global_line - required_on + 1)
                    start_y = global_line - required_on + 1;
                }
            } else {
                on_streak = 0;
            }

            global_line++; // 次の行へ
        }
    }
};


#include <opencv2/opencv.hpp>
#include <numeric>

// マーカ帯の開始点の y を返す。見つからなければ NaN を返す。
// expectedWidthPx: 帯の見かけ短辺（例: 200）
// roi: 画像の一部だけ見たい場合に指定（なければ全体）
float GetMarkerStartY(const cv::Mat& bgr, float expectedWidthPx = 200.0f,
                      const cv::Rect& roi = cv::Rect())
{
    CV_Assert(!bgr.empty());

    // ---- 1) ROI 切り出し ----
    cv::Rect R = roi.area() > 0 ? roi : cv::Rect(0,0,bgr.cols,bgr.rows);
    cv::Mat src = bgr(R).clone();

    // ---- 2) 黒抽出（二値化）+ モルフォロジー ----
    cv::Mat gray; cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(5,5), 1.0);
    cv::Mat bin;
    cv::threshold(gray, bin, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    // 200px級の帯が欠けても繋がる程度にクロージング
    cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9,9));
    cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, k, cv::Point(-1,-1), 1);

    // ---- 3) 輪郭抽出 ----
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return std::numeric_limits<float>::quiet_NaN();

    // ---- 4) minAreaRect で「太い帯」候補を選ぶ ----
    int best = -1; double bestScore = -1;
    const float wMin = expectedWidthPx * 0.8f;   // 200±20% は適宜調整
    const float wMax = expectedWidthPx * 1.2f;
    const float aspectMin = 3.0f;                // 短辺の3倍以上を帯らしさの目安

    for (int i=0;i<(int)contours.size();++i){
        if (contours[i].size() < 20) continue;
        cv::RotatedRect rr = cv::minAreaRect(contours[i]);
        float a = rr.size.width, b = rr.size.height;
        float shortSide = std::min(a,b), longSide = std::max(a,b);
        if (shortSide < wMin || shortSide > wMax) continue;
        if (longSide/shortSide < aspectMin) continue;

        double score = (double)longSide; // より長い帯を優先
        if (score > bestScore){ bestScore = score; best = i; }
    }
    if (best < 0) return std::numeric_limits<float>::quiet_NaN();

    // ---- 5) サブピクセル直線フィット ----
    cv::Vec4f lineParam;
    cv::fitLine(contours[best], lineParam, cv::DIST_L2, 0, 0.01, 0.01);
    cv::Point2f d(lineParam[0], lineParam[1]);     // 方向ベクトル（ほぼ単位）
    cv::Point2f p0(lineParam[2], lineParam[3]);    // 直線上の一点（重心付近）

    // 正規化（fitLine の d は単位長に近いが、厳密に合わせておく）
    float dl = std::sqrt(d.x*d.x + d.y*d.y);
    if (dl < 1e-9f) return std::numeric_limits<float>::quiet_NaN();
    d.x /= dl; d.y /= dl;

    // ---- 6) 輪郭点を直線に射影 → 最小射影値を「開始点」と定義 ----
    // t = (pt - p0)・d    （・は内積）
    float bestT = std::numeric_limits<float>::infinity();
    cv::Point2f bestPt;
    for (auto &pt : contours[best]){
        cv::Point2f fpt((float)pt.x, (float)pt.y);
        cv::Point2f v = fpt - p0;
        float t = v.x*d.x + v.y*d.y;
        if (t < bestT){ bestT = t; bestPt = fpt; }
    }

    // 画像全体座標系の y に戻す（ROI オフセット加算）
    return bestPt.y + (float)R.y;
}




Windows Registry Editor Version 5.00

[HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\YourApp.exe]
"DumpFolder"="C:\\Dumps"
"DumpType"=dword:00000002
"DumpCount"=dword:0000000a


using System;
using System.Drawing;
using System.Windows.Forms;

namespace WinFormsApp
{
    public partial class Form1 : Form
    {
        private Point _lastClickedPointClient = Point.Empty;   // クリック保存用（クライアント座標）

        private readonly ContextMenuStrip _cmenu = new ContextMenuStrip();
        private readonly ToolStripMenuItem _miUsePoint = new ToolStripMenuItem("この座標を入力");

        private readonly PictureBox pictureBox1 = new PictureBox();
        private readonly TextBox textBoxPoint = new TextBox();

        public Form1()
        {
            InitializeComponent();

            // --- PictureBox の初期設定 ---
            pictureBox1.Dock = DockStyle.Fill;
            pictureBox1.BackColor = Color.LightGray;
            pictureBox1.ContextMenuStrip = _cmenu;
            this.Controls.Add(pictureBox1);

            // --- TextBox の初期設定 ---
            textBoxPoint.Dock = DockStyle.Bottom;
            this.Controls.Add(textBoxPoint);

            // --- コンテキストメニュー構築 ---
            _cmenu.Items.Add(_miUsePoint);

            // 右クリック位置を記録
            pictureBox1.MouseDown += (s, e) =>
            {
                if (e.Button == MouseButtons.Right)
                {
                    // pictureBox1 内の相対座標を保存
                    _lastClickedPointClient = e.Location;
                }
            };

            // メニューのコマンドで TextBox に書き込む
            _miUsePoint.Click += (s, e) =>
            {
                textBoxPoint.Text = $"{_lastClickedPointClient.X}, {_lastClickedPointClient.Y}";
            };

            // 念のため、メニューが開く直前に最新座標を再取得
            _cmenu.Opening += (s, e) =>
            {
                if (_cmenu.SourceControl is Control src)
                {
                    var client = src.PointToClient(Cursor.Position);
                    _lastClickedPointClient = client;
                }
            };
        }
    }
}
public class SafeStore
{
    private readonly object _gate = new object(); // 絶対に public にしない & this には lock しない

    private int _value;

    public void Write(int v)
    {
        lock (_gate)
        {
            _value = v;
        }
    }

    public int ReadAndClear()
    {
        lock (_gate)
        {
            int tmp = _value;
            _value = 0;
            return tmp;
        }
    }
}

using var mtx = new Mutex(false, @"Global\MyApp_UniqueResource");
mtx.WaitOne();
try
{
    // 共有資源を使用
}
finally { mtx.ReleaseMutex(); }








// GpuCtx にメンバ追加
cufftHandle planC2C4 = 0;
int tileW = 0, tileH = 0; // プランのサイズを覚えておく


// gp_create_ctx の最後あたり（回転行列などの後）
{
    ctx->tileH = height;
    ctx->tileW = width / 4;                 // W%4==0 前提

    int n[2]      = { ctx->tileW, ctx->tileH };  // {X=幅, Y=高さ}
    int inembed[2]= { ctx->tileW, ctx->tileH };
    int onembed[2]= { ctx->tileW, ctx->tileH };
    int istride   = 1, ostride = 1;
    int idist     = ctx->tileW * ctx->tileH;     // 1スライスの要素数
    int odist     = idist;
    int batch     = 4;

    cufftPlanMany(&ctx->planC2C4, 2, n,
                  inembed, istride, idist,
                  onembed, ostride, odist,
                  CUFFT_C2C, batch);
    // FFTは回転完了後の sFFT ストリームで回す想定（実行時に SetStream）
}

#include "hann_window.hpp"
#include <opencv2/core.hpp>  // createHanningWindow

// GpuCtx 側にキャッシュを持たせる想定
struct GpuCtx {
    std::mutex winMu;
    // 幅だけが変わる運用なら key=W で十分。高さも変わるなら pair<int,int> をキーにする。
    std::unordered_map<int, cv::cuda::GpuMat> winCacheGpu;
};

cv::cuda::GpuMat& getHann2D(GpuCtx* ctx, int H, int W)
{
    std::lock_guard<std::mutex> lk(ctx->winMu);

    if (auto it = ctx->winCacheGpu.find(W); it != ctx->winCacheGpu.end())
        return it->second;

    // CPUで2D Hann生成（OpenCVが外積で作ってくれる）
    cv::Mat hann2d;
    cv::createHanningWindow(hann2d, cv::Size(W, H), CV_32F); // H×W, CV_32FC1

    // GPUへアップロードしてキャッシュ
    cv::cuda::GpuMat winGpu;
    winGpu.upload(hann2d);

    auto [it2, ok] = ctx->winCacheGpu.emplace(W, std::move(winGpu));
    return it2->second;
}

// 2D Hann (H×w) を GPU にキャッシュ（前に出したやつでOK）
cv::cuda::GpuMat& getHann2D(GpuCtx* ctx, int H, int w);

// 実数タイルを float2(Imag=0) へ詰める＋窓を掛けるカーネル
__global__ void pack_with_window(const float* __restrict__ src, size_t srcPitchFloats,
                                 const float* __restrict__ win, // H×w
                                 float2* __restrict__ dst, int w, int H, size_t dstSliceElems)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= w || y >= H) return;
    float v = src[y*srcPitchFloats + x] * win[y*w + x];
    dst[y*w + x] = make_float2(v, 0.0f);
    // dst は各バッチの先頭を渡す想定（呼び出し側で +t*dstSliceElems する）
}


// 2D Hann (H×w) を GPU にキャッシュ（前に出したやつでOK）
cv::cuda::GpuMat& getHann2D(GpuCtx* ctx, int H, int w);

// 実数タイルを float2(Imag=0) へ詰める＋窓を掛けるカーネル
__global__ void pack_with_window(const float* __restrict__ src, size_t srcPitchFloats,
                                 const float* __restrict__ win, // H×w
                                 float2* __restrict__ dst, int w, int H, size_t dstSliceElems)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= w || y >= H) return;
    float v = src[y*srcPitchFloats + x] * win[y*w + x];
    dst[y*w + x] = make_float2(v, 0.0f);
    // dst は各バッチの先頭を渡す想定（呼び出し側で +t*dstSliceElems する）
}

// 2D Hann (H×w) を GPU にキャッシュ（前に出したやつでOK）
cv::cuda::GpuMat& getHann2D(GpuCtx* ctx, int H, int w);

// 実数タイルを float2(Imag=0) へ詰める＋窓を掛けるカーネル
__global__ void pack_with_window(const float* __restrict__ src, size_t srcPitchFloats,
                                 const float* __restrict__ win, // H×w
                                 float2* __restrict__ dst, int w, int H, size_t dstSliceElems)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= w || y >= H) return;
    float v = src[y*srcPitchFloats + x] * win[y*w + x];
    dst[y*w + x] = make_float2(v, 0.0f);
    // dst は各バッチの先頭を渡す想定（呼び出し側で +t*dstSliceElems する）
}

const int H = ctx->H, W = ctx->W;
const int batch = 4;
const int w = ctx->tileW;     // W/4
const size_t sliceElems = (size_t)H * w;

// 連結出力（CV_32FC2, H×W）をGPUに確保
s.d_fft_cat.create(H, W, CV_32FC2);

// batched 入力バッファ（float2, [4][H][w]）を一時確保（スロット再利用推奨）
float2* d_batch = nullptr;
cudaMallocAsync(&d_batch, batch * sliceElems * sizeof(float2),
                cv::cuda::StreamAccessor::getStream(sK));  // sK/sFFTどちらでも

// FFT ストリームを決める（回転に依存）
cv::cuda::Stream sFFT = sK;
sFFT.waitEvent(s.evK);
cufftSetStream(ctx->planC2C4, cv::cuda::StreamAccessor::getStream(sFFT));

// 窓
cv::cuda::GpuMat& win = getHann2D(ctx, H, w);

// 4タイルを pack_with_window で一括パック
dim3 blk(32, 8);
dim3 grd((w + blk.x - 1)/blk.x, (H + blk.y - 1)/blk.y);

for (int t = 0; t < batch; ++t) {
    cv::Rect roi(t * w, 0, w, H);
    cv::cuda::GpuMat tile = s.d_out(roi);     // CV_32FC1
    const float* src = tile.ptr<float>();
    size_t srcPitchF = tile.step / sizeof(float);

    float2* dst = d_batch + (size_t)t * sliceElems;

    pack_with_window<<<grd, blk, 0, cv::cuda::StreamAccessor::getStream(sFFT)>>>(
        src, srcPitchF, win.ptr<float>(), dst, w, H, sliceElems);
}

// batched FFT（in-place）
cufftExecC2C(ctx->planC2C4,
             reinterpret_cast<cufftComplex*>(d_batch),
             reinterpret_cast<cufftComplex*>(d_batch),
             CUFFT_FORWARD);

// batched の各スライスを横に連結して d_fft_cat に配置（GPU内2Dコピー）
for (int t = 0; t < batch; ++t) {
    cv::Rect roi(t * w, 0, w, H);
    // dst: GpuMat ROI (CV_32FC2)
    auto dst = s.d_fft_cat(roi);
    // src: d_batch + t*sliceElems（連続, pitch = w*sizeof(float2)）
    cudaMemcpy2DAsync(dst.ptr(), dst.step,
                      d_batch + (size_t)t * sliceElems, w * sizeof(float2),
                      w * sizeof(float2), H,
                      cudaMemcpyDeviceToDevice,
                      cv::cuda::StreamAccessor::getStream(sFFT));
}

// （ここで d_batch を解放 or 再利用用に Slot に保持）
cudaFreeAsync(d_batch, cv::cuda::StreamAccessor::getStream(sFFT));

// まとめて1回 D2H（CV_32FC2, H×W）
sD2H.waitEvent(s.evK);          // sFFT と同一なら暗黙順序でもOK
s.d_fft_cat.download(s.fft_cat_host, sD2H);  // cv::Mat(CV_32FC2)
cudaEventRecord(s.evD2H, cv::cuda::StreamAccessor::getStream(sD2H));

// D2H 完了 → cudaLaunchHostFunc で軽い通知 → 自前プール or 直接コールバック



// 依存: H2D→回転
sK.waitEvent(s.evH2D);
cv::cuda::warpAffine(s.d_in, s.d_out, ctx->rotM, s.d_out.size(),
                     cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0), sK);
cudaEventRecord(s.evK, cv::cuda::StreamAccessor::getStream(sK));

// ===== ここから窓＋前方DFT（GPU） =====
const int W = ctx->W, H = ctx->H;
const int tiles = 4;
const int baseW = W / tiles;
const int rem   = W % tiles;

// DFT用の出力（複素2ch）をGPUに用意（4帯を横に並べる）
s.d_fft_cat.create(H, W, CV_32FC2);   // <- 新規: Slot に GpuMat 追加しておく

cv::cuda::Stream sFFT = sK;           // 同じでもOK（別にしても可）
sFFT.waitEvent(s.evK);

int x = 0;
for (int t = 0; t < tiles; ++t) {
    int w = baseW + ((t == tiles-1) ? rem : 0);
    cv::Rect roi(x, 0, w, H); x += w;

    // 1) ROI（回転後, CV_32FC1）
    cv::cuda::GpuMat tile = s.d_out(roi);

    // 2) 窓（Hann）をGPUで掛ける
    auto& win = getHann2D(ctx, H, w);
    cv::cuda::multiply(tile, win, tile, 1.0, -1, sFFT); // in-place OK

    // 3) 前方2D DFT（パディングなし, 複素出力 CV_32FC2）
    cv::cuda::GpuMat complex; // H×w×2ch
    cv::cuda::dft(tile, complex, cv::Size(), cv::DFT_COMPLEX_OUTPUT, sFFT);

    // 4) 横に連結（GPU内コピー）
    complex.copyTo(s.d_fft_cat(roi), sFFT);
}

// 5) まとめて1回だけ D2H（複素2chのまま）
sD2H.waitEvent(s.evK); //（sFFT と同一なら暗黙順序でOK）
s.d_fft_cat.download(s.fft_cat_host, sD2H); // cv::Mat(CV_32FC2) を Slot に用意しておく
cudaEventRecord(s.evD2H, cv::cuda::StreamAccessor::getStream(sD2H));

// ===== D2H 完了でホスト関数 → コールバック =====
auto rawD2H = cv::cuda::StreamAccessor::getStream(sD2H);
struct Payload { GpuCtx* ctx; int slot; int fid; void* user; };
auto* p = new Payload{ctx, j.slot, j.frame_id, j.user};

cudaLaunchHostFunc(rawD2H, [](void* ud){
    std::unique_ptr<Payload> P((Payload*)ud);
    auto* ctx = P->ctx; int slot = P->slot; int fid = P->fid; void* user = P->user;
    auto& s = ctx->slots[slot];

    // s.fft_cat_host: CV_32FC2, 幅W×高さH, 各画素が (Re,Im)
    if (ctx->cb) {
        ctx->cb(fid,
                reinterpret_cast<const float*>(s.fft_cat_host.ptr()),
                ctx->W, ctx->H,
                static_cast<int>(s.fft_cat_host.step),   // バイト単位の行ストライド
                user ? user : ctx->user);
    }
    // スロット解放
    { std::lock_guard<std::mutex> lk(ctx->mu);
      s.id = -1; ctx->freeSlots.push(slot); }
    ctx->cv.notify_all();
}, p);



using System;
using System.Diagnostics.Tracing;
using System.Threading;

[EventSource(Name = "MyCompany-MyApp")]
class MyEventSource : EventSource
{
    public static readonly MyEventSource Log = new MyEventSource();

    [Event(1, Message = "FFT開始", Level = EventLevel.Informational)]
    public void FftStart() => WriteEvent(1);

    [Event(2, Message = "FFT終了", Level = EventLevel.Informational)]
    public void FftEnd() => WriteEvent(2);
}

class Program
{
    static void Main()
    {
        MyEventSource.Log.FftStart();
        Thread.Sleep(50); // ダミー処理
        MyEventSource.Log.FftEnd();
    }
}

#include <windows.h>
#include <TraceLoggingProvider.h>

// プロバイダー定義
TRACELOGGING_DEFINE_PROVIDER(
    g_hMyProvider,
    "MyCompany-MyApp",
    // GUID は `uuidgen` で生成
    (0x12345678,0x1234,0x1234,0x12,0x34,0x56,0x78,0x9a,0xbc,0xde,0xf0)
);

int main()
{
    TraceLoggingRegister(g_hMyProvider);

    TraceLoggingWrite(g_hMyProvider, "FFT_Start");
    Sleep(50);
    TraceLoggingWrite(g_hMyProvider, "FFT_End");

    TraceLoggingUnregister(g_hMyProvider);
    return 0;
}




#pragma once
#include <nvToolsExt.h>
#include <string>

class NvtxRange
{
public:
    // スコープ自動管理用 (RAII)
    NvtxRange(const char* name, int category = 0, unsigned int argb = 0xFF80C0FF)
    {
        nvtxEventAttributes_t attr = {};
        attr.version = NVTX_VERSION;
        attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        attr.category = category;
        attr.colorType = NVTX_COLOR_ARGB;
        attr.color = argb;
        attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
        attr.message.ascii = name;
        id_ = nvtxRangeStartEx(&attr);
    }

    ~NvtxRange()
    {
        if (id_ != 0)
            nvtxRangeEnd(id_);
    }

    // 非同期用途：明示的に閉じる
    void End()
    {
        if (id_ != 0)
        {
            nvtxRangeEnd(id_);
            id_ = 0;
        }
    }

    // コピー禁止、ムーブ許可
    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
    NvtxRange(NvtxRange&& other) noexcept { id_ = other.id_; other.id_ = 0; }
    NvtxRange& operator=(NvtxRange&& other) noexcept
    {
        if (this != &other)
        {
            End();
            id_ = other.id_;
            other.id_ = 0;
        }
        return *this;
    }

private:
    nvtxRangeId_t id_{0};
};

using System;
using System.Runtime.InteropServices;

internal static class NvtxEx
{
    [StructLayout(LayoutKind.Sequential)]
    private struct nvtxEventAttributes_t
    {
        public ushort version;
        public ushort size;
        public int category;
        public int colorType;
        public uint color;
        public int messageType;
        public IntPtr message;
    }

    [DllImport("nvToolsExt64_1.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern ulong nvtxRangeStartEx(ref nvtxEventAttributes_t attr);

    [DllImport("nvToolsExt64_1.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern void nvtxRangeEnd(ulong id);

    public static ulong Begin(string name, int cat = 0, uint argb = 0xFF80C0FF)
    {
        var bytes = System.Text.Encoding.ASCII.GetBytes(name + "\0");
        var handle = GCHandle.Alloc(bytes, GCHandleType.Pinned);
        try
        {
            var attr = new nvtxEventAttributes_t
            {
                version = 1,
                size = (ushort)Marshal.SizeOf<nvtxEventAttributes_t>(),
                category = cat,
                colorType = 1,
                color = argb,
                messageType = 1,
                message = handle.AddrOfPinnedObject()
            };
            return nvtxRangeStartEx(ref attr);
        }
        finally
        {
            handle.Free();
        }
    }

    public static void End(ulong id)
    {
        nvtxRangeEnd(id);
    }
}


[StructLayout(LayoutKind.Sequential)]
struct nvtxEventAttributes_t {
    public ushort version; public ushort size;
    public int category; public int colorType; public uint color;
    public int messageType; public IntPtr message; // ANSI
}
internal static class NvtxEx {
    [DllImport("nvToolsExt64_1.dll", CallingConvention = CallingConvention.Cdecl)]
    static extern ulong nvtxRangeStartEx(ref nvtxEventAttributes_t attr);
    [DllImport("nvToolsExt64_1.dll", CallingConvention = CallingConvention.Cdecl)]
    static extern void nvtxRangeEnd(ulong id);

    public static IDisposable Push(string name, int cat = 0, uint rgb = 0x00A0FFFF) {
        var bytes = System.Text.Encoding.ASCII.GetBytes(name + "\0");
        var ptr = Marshal.UnsafeAddrOfPinnedArrayElement(bytes, 0);
        var a = new nvtxEventAttributes_t {
            version = 1, size = (ushort)Marshal.SizeOf<nvtxEventAttributes_t>(),
            category = cat, colorType = 1, color = rgb, messageType = 1, message = ptr
        };
        ulong id = nvtxRangeStartEx(ref a);
        return new Pop{id=id};
    }
    private sealed class Pop : IDisposable { public ulong id; public void Dispose()=>nvtxRangeEnd(id); }
}

using System;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

internal static class Nvtx
{
    // Windows x64 の nvToolsExt
    [DllImport("nvToolsExt64_1.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern int nvtxRangePushA([MarshalAs(UnmanagedType.LPStr)] string message);

    [DllImport("nvToolsExt64_1.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern int nvtxRangePop();

    [DllImport("nvToolsExt64_1.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void nvtxNameOsThread(uint threadId, [MarshalAs(UnmanagedType.LPStr)] string name);

    [DllImport("kernel32.dll")]
    private static extern uint GetCurrentThreadId();

    public static IDisposable Push(string name)
    {
        // スレッドに名前が未設定なら付ける（任意）
        nvtxNameOsThread(GetCurrentThreadId(), $"T{GetCurrentThreadId()}");
        nvtxRangePushA(name);
        return new PopOnDispose();
    }

    private sealed class PopOnDispose : IDisposable
    {
        public void Dispose() => nvtxRangePop();
    }
}

// 使い方例：任意区間をNVTXで囲む
public static class Example
{
    public static void Run()
    {
        Parallel.For(0, 8, i =>
        {
            using (Nvtx.Push($"Item {i}"))
            {
                using (Nvtx.Push("Rotate")) Rotate();
                using (Nvtx.Push("FFT"))    Fft();
                using (Nvtx.Push("Post"))   Post();
            }
        });
    }

    static void Rotate() { /* 対象処理 */ Thread.SpinWait(200000); }
    static void Fft()    { /* 対象処理 */ Thread.SpinWait(400000); }
    static void Post()   { /* 対象処理 */ Thread.SpinWait(150000); }
}

// ComputeBackend.h
#pragma once
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <atomic>
#include <mutex>
#include <string>

enum class ComputeBackend { CPU, GPU };

bool InitComputeBackend();         // 起動時に1回呼ぶ
ComputeBackend GetBackend();       // どっちで動くか取得
void TripToCpu(const char* reason); // 途中でGPUが落ちたらCPUへ切替（回路遮断）
bool IsCudaBuild();                // 参考：ビルドがCUDA対応か

// ComputeBackend.cpp
#include "ComputeBackend.h"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>
#include <cstdlib>
#include <iostream>

namespace {
std::once_flag g_once;
std::atomic<ComputeBackend> g_backend{ComputeBackend::CPU};

bool detectCudaOnce() {
    try {
        // 環境変数で強制CPU（デバッグ運用用）
        if (const char* v = std::getenv("APP_FORCE_CPU")) {
            if (std::string(v) == "1") return false;
        }

        // 1) デバイス数
        int n = cv::cuda::getCudaEnabledDeviceCount();
        if (n <= 0) return false;

        // 2) 互換性（Compute Capabilityなど）
        cv::cuda::DeviceInfo info(0);
        if (!info.isCompatible()) return false;

        // 3) 最小確保テスト（ドライバ異常などの早期検出）
        cv::cuda::GpuMat test(8, 8, CV_8UC1);
        (void)test;

        return true;
    } catch (...) {
        return false;
    }
}
} // namespace

bool InitComputeBackend() {
    std::call_once(g_once, [] {
        g_backend.store(detectCudaOnce() ? ComputeBackend::GPU : ComputeBackend::CPU,
                        std::memory_order_relaxed);
        std::cout << "[Init] ComputeBackend = "
                  << (g_backend.load()==ComputeBackend::GPU ? "GPU" : "CPU") << std::endl;
        if (g_backend.load()==ComputeBackend::GPU) {
            try { cv::cuda::printCudaDeviceInfo(0); } catch (...) {}
        }
    });
    return g_backend.load()==ComputeBackend::GPU;
}

ComputeBackend GetBackend() {
    return g_backend.load(std::memory_order_relaxed);
}

void TripToCpu(const char* reason) {
    auto prev = g_backend.exchange(ComputeBackend::CPU);
    if (prev != ComputeBackend::CPU) {
        std::cerr << "[WARN] Switched to CPU due to GPU failure: "
                  << (reason ? reason : "(unknown)") << std::endl;
    }
}

bool IsCudaBuild() {
    try {
        const auto bi = cv::getBuildInformation();
        return bi.find("CUDA") != std::string::npos; // 参考表示用（判定は detectCudaOnce が本体）
    } catch (...) {
        return false;
    }
}



// 依存: 回転完了
cv::cuda::Stream sFFT = sK;
sFFT.waitEvent(s.evK);

const int W = ctx->W, H = ctx->H;
const int tiles = 4;
const int baseW = W / tiles;
const int rem   = W % tiles;

s.d_mag_cat.create(H, W, CV_32FC1);

int x = 0;
for (int t = 0; t < tiles; ++t) {
    int w = baseW + ((t == tiles-1) ? rem : 0);
    cv::Rect roi(x, 0, w, H);
    x += w;

    cv::cuda::GpuMat tile = s.d_out(roi); // 回転後のタイル CV_32FC1

    // ★ ここで窓を掛ける（GPU）
    cv::cuda::GpuMat& winGpu = getHann2D(ctx, H, w);
    cv::cuda::multiply(tile, winGpu, tile, 1.0, -1, sFFT); // tile ← tile * window

    // 2D FFT（複素出力, パディングなし）
    cv::cuda::GpuMat complex; // CV_32FC2
    cv::cuda::dft(tile, complex, cv::Size(), cv::DFT_COMPLEX_OUTPUT, sFFT);

    // magnitude
    std::vector<cv::cuda::GpuMat> planes;
    cv::cuda::split(complex, planes, sFFT);
    cv::cuda::GpuMat mag;
    cv::cuda::magnitude(planes[0], planes[1], mag, sFFT);

    // 横に連結（GPU内）
    mag.copyTo(s.d_mag_cat(roi), sFFT);
}

// まとめて1回 D2H → Host コールバック（前回と同じ）



// gpu_async_fftpool.cpp  — ① 非同期Submit + DLL内 FFT専用スレッドプール + 最終結果コールバック
// ビルド例: cl /O2 /MD /EHsc /LD gpu_async_fftpool.cpp /I<opencv\include> <opencv libs...>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#ifdef _WIN32
  #define DLL_EXPORT __declspec(dllexport)
  #define DLL_CALL   __cdecl
#else
  #define DLL_EXPORT
  #define DLL_CALL
#endif

extern "C" {
// 最終結果（回転→FFTのmagnitude）だけを返す
typedef void (DLL_CALL *ResultCallback)(
    int frameId,
    const float* data,   // 先頭ポインタ（row0）
    int width, int height,
    int strideBytes,     // 1行のバイト数（= width*sizeof(float) が基本）
    void* user           // gp_submit_* の引数 user をそのまま返す
);

// 不透明ハンドル
struct GpuCtx;

// コンテキスト作成/破棄
DLL_EXPORT GpuCtx* DLL_CALL gp_create_ctx(int width, int height, int nbuf,
                                          float angle_deg,
                                          ResultCallback cb, void* user_global);
DLL_EXPORT void    DLL_CALL gp_destroy_ctx(GpuCtx* ctx);

// 非ブロッキング: 空きが無ければ -2
DLL_EXPORT int     DLL_CALL gp_submit_try (GpuCtx* ctx, int frameId,
                                           const uint8_t* src, int pitchBytes,
                                           void* user_per_job);

// ブロッキング: 空きを待つ。timeout_ms<=0 なら無期限。タイムアウトで -2
DLL_EXPORT int     DLL_CALL gp_submit_wait(GpuCtx* ctx, int frameId,
                                           const uint8_t* src, int pitchBytes,
                                           void* user_per_job, int timeout_ms);
} // extern "C"

//================ 内部実装 =================//

// ---- 簡易固定スレッドプール（FFT用） ----
class ThreadPool {
public:
    explicit ThreadPool(size_t n){ start(n); }
    ~ThreadPool(){ stop(); }

    template<class F>
    void submit(F&& f){
        {
            std::lock_guard<std::mutex> lk(mu_);
            q_.emplace(std::function<void()>(std::forward<F>(f)));
        }
        cv_.notify_one();
    }

private:
    void start(size_t n){
        if (n<1) n=1;
        for(size_t i=0;i<n;++i){
            ws_.emplace_back([this]{
                for(;;){
                    std::function<void()> job;
                    {
                        std::unique_lock<std::mutex> lk(mu_);
                        cv_.wait(lk,[&]{ return stop_ || !q_.empty(); });
                        if (stop_ && q_.empty()) return;
                        job = std::move(q_.front()); q_.pop();
                    }
                    try { job(); } catch(...) { /* log等 */ }
                }
            });
        }
    }
    void stop(){
        { std::lock_guard<std::mutex> lk(mu_); stop_=true; }
        cv_.notify_all();
        for (auto& t: ws_) if (t.joinable()) t.join();
    }

    std::vector<std::thread> ws_;
    std::queue<std::function<void()>> q_;
    std::mutex mu_; std::condition_variable cv_;
    bool stop_ = false;
};

// ---- Job / Slot ----
struct Job {
    int frame_id;
    int slot;
    void* user; // per-submit の user
};

struct Slot {
    int id = -1; // -1:空き, -2:予約, >=0:frameId（使用中）

    // ホスト(Pinned) と GPU
    cv::cuda::HostMem pin_in, pin_rot;
    cv::Mat in_mat, rot_mat;          // rot_mat: 回転後 float32 1ch
    cv::cuda::GpuMat d_in, d_rot;

    // CUDA Events
    cudaEvent_t evH2D=nullptr, evK=nullptr, evD2H=nullptr;

    // FFTワーク
    cv::Mat fft_complex, fft_mag;     // CV_32FC2 / CV_32FC1
};

// ---- コンテキスト ----
struct GpuCtx {
    int W=0, H=0, N=0;
    cv::Mat rotM;
    std::vector<Slot> slots;

    // 単一のロック/条件変数（シンプル派）
    std::mutex mu;
    std::condition_variable cv;

    // スロット空き管理 & ジョブキュー
    std::queue<int> freeSlots;   // 空きslot番号
    std::queue<Job> jobQueue;    // GPUワーカー行き

    // 終了フラグ／ワーカー
    bool quitting = false;
    std::thread gpuWorker;

    // CUDA ストリーム
    cv::cuda::Stream sH2D, sK, sD2H;

    // FFT 専用スレッドプール
    std::unique_ptr<ThreadPool> fftPool;

    // コールバック
    ResultCallback cb = nullptr;
    void* user_global = nullptr;
};

// ---- ヘルパ ----
static void make_hostmat(cv::cuda::HostMem& hm, int h, int w, int type, cv::Mat& header){
    hm.release();
    hm = cv::cuda::HostMem(h, w, type, cv::cuda::HostMem::PAGE_LOCKED);
    header = hm.createMatHeader();
}

// ---- GPUワーカー ----
static void gpu_worker_loop(GpuCtx* ctx){
    auto& sH2D = ctx->sH2D;
    auto& sK   = ctx->sK;
    auto& sD2H = ctx->sD2H;

    while (true){
        Job j;
        {
            std::unique_lock<std::mutex> lk(ctx->mu);
            ctx->cv.wait(lk, [&]{ return ctx->quitting || !ctx->jobQueue.empty(); });
            if (ctx->quitting && ctx->jobQueue.empty()) break;
            j = ctx->jobQueue.front(); ctx->jobQueue.pop();
        }

        // ここからは slot を専有
        auto& sl = ctx->slots[j.slot];
        sl.id = j.frame_id; // 予約(-2) → 実使用(frameId)

        // 1) H2D
        sl.d_in.upload(sl.in_mat, sH2D);
        cudaEventRecord(sl.evH2D, cv::cuda::StreamAccessor::getStream(sH2D));

        // 2) 回転（H2Dに依存）
        sK.waitEvent(sl.evH2D);
        cv::cuda::warpAffine(sl.d_in, sl.d_rot, ctx->rotM, sl.d_rot.size(),
                             cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0), sK);
        cudaEventRecord(sl.evK, cv::cuda::StreamAccessor::getStream(sK));

        // 3) D2H（Kernelに依存）
        sD2H.waitEvent(sl.evK);
        sl.d_rot.download(sl.rot_mat, sD2H);
        cudaEventRecord(sl.evD2H, cv::cuda::StreamAccessor::getStream(sD2H));

        // D2H 完了を同期（ここで GPU は手離れ）
        cudaEventSynchronize(sl.evD2H);

        // 4) FFT は“プールへ投げる” → GPUワーカーは即次ジョブへ
        GpuCtx* ctx2 = ctx;
        int slot_idx = j.slot;
        int fid = j.frame_id;
        void* user = j.user;

        ctx->fftPool->submit([ctx2, slot_idx, fid, user]{
            auto& s = ctx2->slots[slot_idx];

            // （任意）過剰並列回避：OpenCV内部スレッドをOFF
            // cv::setNumThreads(1);

            // FFT（複素）→ magnitude
            cv::dft(s.rot_mat, s.fft_complex, cv::DFT_COMPLEX_OUTPUT);
            cv::Mat planes[2];
            cv::split(s.fft_complex, planes);
            cv::magnitude(planes[0], planes[1], s.fft_mag); // CV_32FC1

            // コールバック（最終結果のみ）
            if (ctx2->cb){
                ctx2->cb(fid,
                         reinterpret_cast<const float*>(s.fft_mag.ptr()),
                         ctx2->W, ctx2->H,
                         static_cast<int>(s.fft_mag.step),
                         user ? user : ctx2->user_global);
            }

            // スロット解放 → 空き通知
            {
                std::lock_guard<std::mutex> lk(ctx2->mu);
                s.id = -1;
                ctx2->freeSlots.push(slot_idx);
            }
            ctx2->cv.notify_all();
        });
    }
}

//================ 公開API =================//

extern "C" {

DLL_EXPORT GpuCtx* DLL_CALL gp_create_ctx(int width, int height, int nbuf,
                                          float angle_deg,
                                          ResultCallback cb, void* user_global)
{
    if (nbuf < 2) nbuf = 2;

    auto ctx = new GpuCtx();
    ctx->W = width; ctx->H = height; ctx->N = nbuf;
    ctx->cb = cb; ctx->user_global = user_global;

    // 回転行列
    cv::Point2f c(width/2.f, height/2.f);
    ctx->rotM = cv::getRotationMatrix2D(c, angle_deg, 1.0);

    // スロット確保（Pinned/GPU/Events）
    ctx->slots.resize(nbuf);
    for (int i=0;i<nbuf;++i){
        auto& s = ctx->slots[i];
        s.id = -1;
        make_hostmat(s.pin_in,  height, width, CV_8UC1,  s.in_mat);
        make_hostmat(s.pin_rot, height, width, CV_32FC1, s.rot_mat);
        s.d_in .create(height, width, CV_8UC1);
        s.d_rot.create(height, width, CV_32FC1);
        cudaEventCreateWithFlags(&s.evH2D, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&s.evK,   cudaEventDisableTiming);
        cudaEventCreateWithFlags(&s.evD2H, cudaEventDisableTiming);

        ctx->freeSlots.push(i); // 全部空き
    }

    // FFT プール起動（物理コアに合わせ調整）
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    unsigned nfft = std::max(2u, hw/2); // まずは物理コアの半分
    ctx->fftPool = std::make_unique<ThreadPool>(nfft);

    // GPUワーカー起動（1本でOK：複数ストリームで重ねる）
    ctx->gpuWorker = std::thread(gpu_worker_loop, ctx);
    return ctx;
}

DLL_EXPORT void DLL_CALL gp_destroy_ctx(GpuCtx* ctx){
    if (!ctx) return;

    // submit待ち/worker待ちを起こす
    {
        std::lock_guard<std::mutex> lk(ctx->mu);
        ctx->quitting = true;
    }
    ctx->cv.notify_all();

    if (ctx->gpuWorker.joinable()) ctx->gpuWorker.join();

    // FFTプール停止
    ctx->fftPool.reset();

    // CUDAリソース解放
    for (auto& s : ctx->slots){
        if (s.evH2D) cudaEventDestroy(s.evH2D);
        if (s.evK)   cudaEventDestroy(s.evK);
        if (s.evD2H) cudaEventDestroy(s.evD2H);
    }
    delete ctx;
}

// 非ブロッキング（空き無しなら -2）
DLL_EXPORT int DLL_CALL gp_submit_try(GpuCtx* ctx, int frameId,
                                      const uint8_t* src, int pitchBytes,
                                      void* user_per_job)
{
    if (!ctx || !src) return -1;

    int slot = -1;
    {
        std::lock_guard<std::mutex> lk(ctx->mu);
        if (ctx->freeSlots.empty()) return -2;
        slot = ctx->freeSlots.front(); ctx->freeSlots.pop();
        ctx->slots[slot].id = -2; // 予約
    }

    // 入力を Pinned に即コピー（呼び出し側寿命から解放）
    auto& s = ctx->slots[slot];
    for (int y=0; y<ctx->H; ++y){
        std::memcpy(s.in_mat.ptr(y), src + y*pitchBytes, ctx->W);
    }

    // Job をキューへ → worker 起こす
    {
        std::lock_guard<std::mutex> lk(ctx->mu);
        ctx->jobQueue.push(Job{frameId, slot, user_per_job});
    }
    ctx->cv.notify_all();
    return 0;
}

// ブロッキング（空きが出るまで待つ／タイムアウトあり）
DLL_EXPORT int DLL_CALL gp_submit_wait(GpuCtx* ctx, int frameId,
                                       const uint8_t* src, int pitchBytes,
                                       void* user_per_job, int timeout_ms)
{
    if (!ctx || !src) return -1;

    int slot = -1;
    {
        std::unique_lock<std::mutex> lk(ctx->mu);
        auto pred = [&]{ return ctx->quitting || !ctx->freeSlots.empty(); };
        if (timeout_ms <= 0) {
            ctx->cv.wait(lk, pred);
            if (ctx->quitting) return -3;
        } else {
            if (!ctx->cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), pred))
                return -2; // タイムアウト
            if (ctx->quitting) return -3;
        }
        slot = ctx->freeSlots.front(); ctx->freeSlots.pop();
        ctx->slots[slot].id = -2; // 予約
    }

    auto& s = ctx->slots[slot];
    for (int y=0; y<ctx->H; ++y){
        std::memcpy(s.in_mat.ptr(y), src + y*pitchBytes, ctx->W);
    }

    {
        std::lock_guard<std::mutex> lk(ctx->mu);
        ctx->jobQueue.push(Job{frameId, slot, user_per_job});
    }
    ctx->cv.notify_all();
    return 0;
}

} // extern "C"