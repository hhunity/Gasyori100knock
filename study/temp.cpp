#include <httplib.h>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <memory>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>

// ===== SSE Session / Hub =====
struct Session {
  std::mutex m;
  std::condition_variable cv;
  std::deque<std::string> q;
  bool closed = false;

  void push(std::string msg) {
    {
      std::lock_guard lk(m);
      q.push_back(std::move(msg));
    }
    cv.notify_one();
  }
};

struct Hub {
  std::mutex m;
  std::vector<std::weak_ptr<Session>> subs;

  void subscribe(const std::shared_ptr<Session>& s) {
    std::lock_guard lk(m);
    subs.emplace_back(s);
  }

  void broadcast(const std::string& msg) {
    std::lock_guard lk(m);
    subs.erase(std::remove_if(subs.begin(), subs.end(),
      [&](std::weak_ptr<Session>& w){
        if (auto s = w.lock()) { s->push(msg); return false; }
        return true;
      }), subs.end());
  }
};

// ===== JSON builder (手書き最小) =====
// points = vector<pair<x,y>> を [{"x":..,"y":..}, ...] にする
static std::string make_points_json(const std::vector<std::pair<long long, double>>& points) {
  std::ostringstream os;
  os << "{\"points\":[";
  for (size_t i = 0; i < points.size(); ++i) {
    if (i) os << ",";
    os << "{\"x\":" << points[i].first << ",\"y\":" << points[i].second << "}";
  }
  os << "]}";
  return os.str();
}

static std::string sse_event(const std::string& event, const std::string& data_json) {
  return "event: " + event + "\n"
         "data: " + data_json + "\n\n";
}

// ===== 画像処理が終わったら呼ぶ想定の関数 =====
static void publish_batch(Hub& hub,
                          const std::vector<std::pair<long long, double>>& points)
{
  // event名は "batch" にして、JS側は addEventListener('batch', ...) で受ける
  hub.broadcast(sse_event("batch", make_points_json(points)));
}

int main() {
  httplib::Server svr;
  Hub hub;

  // SSE購読
  svr.Get("/sse", [&](const httplib::Request&, httplib::Response& res) {
    res.set_header("Content-Type", "text/event-stream");
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");

    auto session = std::make_shared<Session>();
    hub.subscribe(session);

    res.set_chunked_content_provider(
      "text/event-stream",
      [session](size_t, httplib::DataSink& sink) {
        std::unique_lock lk(session->m);
        session->cv.wait(lk, [&]{ return session->closed || !session->q.empty(); });
        if (session->closed) return false;

        std::string chunk;
        while (!session->q.empty()) {
          chunk += std::move(session->q.front());
          session->q.pop_front();
        }
        lk.unlock();

        return sink.write(chunk.data(), chunk.size());
      },
      [session](bool) {
        std::lock_guard lk(session->m);
        session->closed = true;
        session->cv.notify_one();
      }
    );
  });

  // デモ用：/emit_batch を叩いたら「複数点」をまとめて配信
  svr.Post("/emit_batch", [&](const httplib::Request&, httplib::Response& res) {
    std::vector<std::pair<long long, double>> pts = {
      {1710000000000LL, 10.0},
      {1710000000100LL, 12.5},
      {1710000000200LL, 11.8}
    };
    publish_batch(hub, pts);
    res.set_content("ok", "text/plain");
  });

  svr.listen("0.0.0.0", 8080);
}

res.set_chunked_content_provider(
  "text/event-stream",
  [session](size_t, httplib::DataSink& sink) {
    std::unique_lock lk(session->m);

    // ★ 10秒ごとにタイムアウト
    session->cv.wait_for(lk, std::chrono::seconds(10), [&]{
      return session->closed || !session->q.empty();
    });

    if (session->closed) return false;

    std::string chunk;

    if (session->q.empty()) {
      // ★ heartbeat
      chunk = ": keep-alive\n\n";
    } else {
      while (!session->q.empty()) {
        chunk += std::move(session->q.front());
        session->q.pop_front();
      }
    }

    lk.unlock();
    return sink.write(chunk.data(), chunk.size());
  },