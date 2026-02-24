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
