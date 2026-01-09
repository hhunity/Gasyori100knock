//c#でおくる
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;

// 例: params_json + config_file を同一POST
static async Task StartCameraAsync(string url, string configFilePath)
{
    using var http = new HttpClient();

    // JSON（文字列）
    var json = """
    {
      "camera_id": "cam01",
      "exposure_us": 2000,
      "gain": 3.2,
      "mode": "HDR"
    }
    """;

    using var form = new MultipartFormDataContent();

    // JSONフィールド（テキスト）
    var jsonPart = new StringContent(json, Encoding.UTF8, "application/json");
    form.Add(jsonPart, "params_json"); // サーバ側のフィールド名と一致させる

    // ファイルフィールド
    var fileBytes = await File.ReadAllBytesAsync(configFilePath);
    var filePart = new ByteArrayContent(fileBytes);
    filePart.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");

    // 重要: Add(content, name, fileName) の fileName が multipart の filename になる
    form.Add(filePart, "config_file", Path.GetFileName(configFilePath));

    // POST
    using var res = await http.PostAsync(url, form);
    var body = await res.Content.ReadAsStringAsync();

    res.EnsureSuccessStatusCode();
    Console.WriteLine(body);
}


//複数ファイル受け取り
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <fstream>

int main() {
  httplib::Server svr;

  svr.Post("/camera/start", [](const httplib::Request& req, httplib::Response& res) {
    // 1) JSON（テキスト）
    if (!req.form.has_field("params_json")) {
      res.status = 400;
      res.set_content("missing params_json", "text/plain");
      return;
    }
    const auto json_str = req.form.get_field("params_json");
    nlohmann::json j;
    try {
      j = nlohmann::json::parse(json_str);
    } catch (...) {
      res.status = 400;
      res.set_content("invalid json", "text/plain");
      return;
    }

    // 2) ファイル（1個）
    if (!req.form.has_file("config_file")) {
      res.status = 400;
      res.set_content("missing config_file", "text/plain");
      return;
    }
    const auto f = req.form.get_file_value("config_file"); // 単体

    // 保存するなら
    {
      std::ofstream ofs(("recv_" + f.filename).c_str(), std::ios::binary);
      ofs.write(f.content.data(), static_cast<std::streamsize>(f.content.size()));
    }

    // 3) JSONの中身で起動（例）
    // start_camera( CameraConfig{...}, f.content ... );

    res.set_content("OK", "text/plain");
  });

  svr.listen("0.0.0.0", 8080);
}





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