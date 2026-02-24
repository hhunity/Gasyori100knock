
#include <iostream>
#include "sapclassbasic.h"

// ---------------------------------------------------------
// 画像が1フレーム（指定ライン数）取得されるたびに呼ばれるコールバック
// ---------------------------------------------------------
void XferCallback(SapXferCallbackInfo *pInfo) {
    if (pInfo->IsTrash()) {
        std::cout << "フレームが破棄されました（処理落ち等の可能性）" << std::endl;
        return;
    }

    // コールバックが呼ばれた回数（取得フレーム数）をカウント
    static int frameCount = 0;
    frameCount++;
    
    std::cout << "連続フレーム取得: " << frameCount << " 枚目" << std::endl;

    // ※ ここで pInfo->GetSapBuffer() から画像データを取り出して処理します
}

int main() {
    // ---------------------------------------------------------
    // 1. 設定ファイルのパスと接続先の指定
    // ---------------------------------------------------------
    // CamExpertで保存した設定ファイル(.ccf)の絶対パスを指定します
    const char* configFilePath = "C:\\path\\to\\your\\camera_config.ccf";

    // 接続先（CamExpertの左ツリーで表示されるサーバー名）
    SapLocation loc("GigEVision_1", 0);

    // ---------------------------------------------------------
    // 2. Saperaオブジェクトの宣言（コンフィグファイルを渡す）
    // ---------------------------------------------------------
    // 第2引数にコンフィグファイルのパスを渡すことで設定をロードします
    SapAcqDevice acq(&loc, configFilePath);
    
    // ※ CameraLinkやCoaXPressなどのフレームグラバー経由の場合は以下を使います
    // SapAcquisition acq(&loc, configFilePath);

    // 連続取り込み用のリングバッファの構築（ここでは10枚分のバッファを確保）
    // カメラから次々来る画像を10枚のメモリで使い回します
    SapBuffer buffers(10, &acq);
    
    // 転送オブジェクト（コールバックの登録）
    SapAcqDeviceToBuf xfer(&acq, &buffers, XferCallback, nullptr);
    // ※ フレームグラバーの場合は以下
    // SapAcqToBuf xfer(&acq, &buffers, XferCallback, nullptr);

    // ---------------------------------------------------------
    // 3. 各オブジェクトの生成（デバイス接続とメモリ確保）
    // ---------------------------------------------------------
    std::cout << "デバイスに接続中..." << std::endl;
    if (!acq.Create()) {
        std::cerr << "カメラの初期化に失敗しました。CCFファイルのパスを確認してください。" << std::endl;
        return -1;
    }
    if (!buffers.Create()) {
        std::cerr << "バッファの初期化に失敗しました。" << std::endl;
        acq.Destroy(); return -1;
    }
    if (!xfer.Create()) {
        std::cerr << "転送オブジェクトの初期化に失敗しました。" << std::endl;
        buffers.Destroy(); acq.Destroy(); return -1;
    }

    // ---------------------------------------------------------
    // 4. 連続画像取得（Grab）の開始
    // ---------------------------------------------------------
    std::cout << "連続取得を開始します。Enterキーで終了します..." << std::endl;
    
    // Grab()を呼ぶとバックグラウンドで連続撮影が始まり、コールバックが呼ばれ続けます
    xfer.Grab(); 

    // ユーザー入力（Enterキー）待ち
    std::cin.get();

    // ---------------------------------------------------------
    // 5. 終了処理
    // ---------------------------------------------------------
    std::cout << "停止処理中..." << std::endl;
    xfer.Freeze();      // 連続転送の停止要求
    xfer.Wait(1000);    // 完全に停止するまで待機

    xfer.Destroy();
    buffers.Destroy();
    acq.Destroy();

    std::cout << "プログラムを終了します。" << std::endl;
    return 0;
}




#include <iostream>
// Sapera LTの基本ヘッダ
#include "sapclassbasic.h"

// ---------------------------------------------------------
// 画像が1フレーム取得されるたびに呼ばれるコールバック関数
// ---------------------------------------------------------
void XferCallback(SapXferCallbackInfo *pInfo) {
    // エラーチェック
    if (pInfo->IsTrash()) {
        std::cout << "フレームが破棄されました（Trash）" << std::endl;
        return;
    }

    std::cout << "フレームを取得しました！" << std::endl;
    // ※ ここで pInfo->GetSapBuffer() を使って画像データ（ポインタ）にアクセスし、
    // OpenCVのMatに変換するなどの画像処理を行います。
}

int main() {
    // 1. 接続先（ロケーション）の指定
    // 第1引数はSapera Configurationツールで設定されているサーバー名（"GigEVision_1" や "CameraLink_1" など）
    SapLocation loc("GigEVision_1", 0);

    // 2. Saperaオブジェクトの宣言
    // GigEカメラの場合は SapAcqDevice を使用。
    // （※フレームグラバー経由の場合は SapAcquisition に変更してください）
    SapAcqDevice acq(&loc);
    
    // バッファ（画像メモリ）の宣言。ここではカメラの出力フォーマットに合わせて1枚分のバッファを用意
    SapBuffer buffers(1, &acq);
    
    // 転送オブジェクトの宣言（カメラからバッファへ転送し、コールバックを呼ぶ）
    SapAcqDeviceToBuf xfer(&acq, &buffers, XferCallback, nullptr);

    // 3. 各オブジェクトの生成（メモリ確保やデバイスとの実際の接続）
    std::cout << "デバイスに接続中..." << std::endl;
    if (!acq.Create()) {
        std::cerr << "カメラの初期化に失敗しました。" << std::endl;
        return -1;
    }
    if (!buffers.Create()) {
        std::cerr << "バッファの初期化に失敗しました。" << std::endl;
        acq.Destroy();
        return -1;
    }
    if (!xfer.Create()) {
        std::cerr << "転送オブジェクトの初期化に失敗しました。" << std::endl;
        buffers.Destroy();
        acq.Destroy();
        return -1;
    }

    // 4. 画像取得（Grab）の開始
    std::cout << "画像取得を開始します。Enterキーで終了します..." << std::endl;
    xfer.Grab();

    // ユーザー入力（Enterキー）待ち
    std::cin.get();

    // 5. 終了処理（安全に停止してメモリを解放）
    std::cout << "停止処理中..." << std::endl;
    xfer.Freeze();      // 転送の停止要求
    xfer.Wait(1000);    // 完全に停止するまで最大1秒待機

    // 作成した逆順で破棄
    xfer.Destroy();
    buffers.Destroy();
    acq.Destroy();

    std::cout << "プログラムを終了します。" << std::endl;
    return 0;
}



factry

std::unique_ptr<ICameraDriver> CreateDriver(const CameraSpec& spec) {
  switch (spec.vendor) {
    case Vendor::VendorA: return std::make_unique<VendorADriver>(/*spec.device_id*/);
    case Vendor::VendorB: return std::make_unique<VendorBDriver>(/*...*/);
    default: throw std::runtime_error("Unknown vendor");
  }
}

class CameraController {
public:
  explicit CameraController(std::unique_ptr<ICameraDriver> drv)
    : drv_(std::move(drv)) {}

  void Open()  { drv_->Open(); }
  void Close() { drv_->Close(); }

  void Configure(const CameraConfig& cfg) { drv_->Apply(cfg); }

  void Start() { drv_->Start(); }
  void Stop()  { drv_->Stop(); }

  bool Poll(FrameView& f) { return drv_->TryGetFrame(f); }

private:
  std::unique_ptr<ICameraDriver> drv_;
};

struct CameraConfig {
  int width = 0, height = 0;
  int offset_x = 0, offset_y = 0;
  double exposure_us = 0;
  double gain_db = 0;
  bool external_trigger = false;
  // ここは「標準化できる最低限」だけ
};

struct FrameView {
  const uint8_t* data = nullptr;
  int width = 0, height = 0, stride = 0;
  uint64_t timestamp_ns = 0;
  uint64_t frame_id = 0;
  // バッファ寿命はshared_ptr等で別途管理するのが安全
};

enum class Feature { Trigger, Strobe, ChunkData, Temperature, LineRate };

class ICameraDriver {
public:
  virtual ~ICameraDriver() = default;

  virtual void Open() = 0;
  virtual void Close() = 0;

  virtual void Apply(const CameraConfig& cfg) = 0;

  virtual void Start() = 0;
  virtual void Stop() = 0;

  // 取得方式はここで統一（Controller側の設計に合わせる）
  virtual bool TryGetFrame(FrameView& out) = 0;

  virtual bool Has(Feature f) const = 0;

  // 拡張: SDK差が大きいところは property で逃がす
  virtual bool TrySet(const std::string& key, const std::string& value) = 0;
  virtual bool TrySet(const std::string& key, double value) = 0;
};
