
//
//PipelineConfig は「外部入力の正規化結果」として集約してよい
//ただし構築時に分解する
auto cfg = load_pipeline_config("config.json"); // JSONからパースして型に落とす

auto cam  = std::make_unique<Camera>(cfg.camera);
auto proc = std::make_unique<ImageProcessor>(cfg.processing);

Pipeline pl(*cam, *proc); // pipeline は非所有参照（前の話）

auto cfg = load_pipeline_config("config.json"); // JSONからパースして型に落とす

auto cam  = std::make_unique<Camera>(cfg.camera);
auto proc = std::make_unique<ImageProcessor>(cfg.processing);

Pipeline pl(*cam, *proc); // pipeline は非所有参照（前の話）

//--------------------------------------------

class Camera {
public:
    void start();
    void stop();
    // grab(), registerCallback() etc...
};

class ImageProcessor {
public:
    void start();
    void stop();
    // process(frame) etc...
};

// “所有しない” パイプライン：部品は参照で受ける
class Pipeline {
public:
    Pipeline(Camera& cam, ImageProcessor& proc)
        : cam_(cam), proc_(proc) {}

    void start_full() {
        cam_.start();
        proc_.start();
        // tbb task group start etc...
    }

    void stop_full() {
        // stop tasks first, then proc, then cam
        proc_.stop();
        cam_.stop();
    }

private:
    Camera& cam_;
    ImageProcessor& proc_;
};

class AppController {
public:
    AppController()
      : cam_(std::make_unique<Camera>()),
        proc_(std::make_unique<ImageProcessor>()),
        pipeline_(std::make_unique<Pipeline>(*cam_, *proc_)) {}

    void start_camera_only() {
        cam_->start();
    }

    void start_everything() {
        pipeline_->start_full();
    }

    void stop_all() {
        pipeline_->stop_full();
    }

private:
    std::unique_ptr<Camera> cam_;
    std::unique_ptr<ImageProcessor> proc_;
    std::unique_ptr<Pipeline> pipeline_;
};