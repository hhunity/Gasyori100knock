#include <httplib.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <memory>

static std::atomic<long long> g_id{0};

static std::string make_sse(const char* event_name, const std::string& data_json) {
    long long id = ++g_id;
    std::string out;
    out += "event: "; out += event_name; out += "\n";
    out += "id: ";    out += std::to_string(id); out += "\n";
    out += "data: ";  out += data_json; out += "\n\n";
    return out;
}

struct Subscriber {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::string> q;
    bool closed = false;

    void push(std::string msg) {
        {
            std::lock_guard<std::mutex> lk(m);
            q.emplace_back(std::move(msg));
        }
        cv.notify_one();
    }

    // timeout で待つ。closedなら false
    bool pop_wait(std::string& out, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lk(m);
        cv.wait_for(lk, timeout, [&] { return closed || !q.empty(); });
        if (closed) return false;
        if (q.empty()) { out.clear(); return true; } // timeout
        out = std::move(q.front());
        q.pop_front();
        return true;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lk(m);
            closed = true;
        }
        cv.notify_all();
    }
};

class SseBroker {
public:
    std::shared_ptr<Subscriber> add() {
        auto sub = std::make_shared<Subscriber>();
        {
            std::lock_guard<std::mutex> lk(m_);
            subs_.push_back(sub);
        }
        return sub;
    }

    void publish(const std::string& msg) {
        std::lock_guard<std::mutex> lk(m_);
        auto it = subs_.begin();
        while (it != subs_.end()) {
            if (auto s = it->lock()) {
                s->push(msg);
                ++it;
            } else {
                it = subs_.erase(it);
            }
        }
    }

private:
    std::mutex m_;
    std::vector<std::weak_ptr<Subscriber>> subs_;
};

static SseBroker g_broker;

static std::atomic<bool> g_running{false};
static std::atomic<bool> g_cancel{false};

int main() {
    httplib::Server svr;

    // SSEは接続ごとにスレッドを占有するので増やす
    svr.new_task_queue = [] { return new httplib::ThreadPool(16); };

    svr.Get("/events", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Content-Type", "text/event-stream; charset=utf-8");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        // res.set_header("X-Accel-Buffering", "no"); // nginx等がいるなら有効なことがある

        auto sub = g_broker.add();

        // 初回メッセージ（この接続だけに）
        sub->push(make_sse("hello", R"({"msg":"connected"})"));

        res.set_chunked_content_provider(
            "text/event-stream",
            [sub](size_t, httplib::DataSink& sink) {
                const auto ping_interval = std::chrono::milliseconds(15000);

                // ヘッダを確実に流すために、まずコメントを1回書く（重要）
                {
                    const char* pre = ": connected\n\n";
                    if (!sink.write(pre, std::strlen(pre))) {
                        sink.done();
                        return true;
                    }
                }

                while (true) {
                    std::string msg;
                    if (!sub->pop_wait(msg, ping_interval)) break; // closed

                    if (!msg.empty()) {
                        if (!sink.write(msg.data(), msg.size())) break;
                    } else {
                        const char* ping = ": ping\n\n";
                        if (!sink.write(ping, std::strlen(ping))) break;
                    }
                }

                sub->close();
                sink.done();
                return true;
            }
        );
    });

    svr.Post("/start", [](const httplib::Request&, httplib::Response& res) {
        if (g_running.exchange(true)) {
            res.set_content(R"({"ok":false,"reason":"already running"})", "application/json");
            return;
        }

        g_cancel.store(false);
        g_broker.publish(make_sse("status", R"({"state":"started"})"));

        std::thread([] {
            for (int p = 0; p <= 100; p += 10) {
                if (g_cancel.load()) {
                    g_broker.publish(make_sse("status", R"({"state":"canceled"})"));
                    g_running.store(false);
                    return;
                }
                g_broker.publish(make_sse("progress", std::string(R"({"pct":)") + std::to_string(p) + "}"));
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
            g_broker.publish(make_sse("complete", R"({"result":"OK"})"));
            g_running.store(false);
        }).detach();

        res.set_content(R"({"ok":true})", "application/json");
    });

    svr.Post("/reset", [](const httplib::Request&, httplib::Response& res) {
        g_cancel.store(true);
        g_broker.publish(make_sse("reset", R"({"msg":"reset requested"})"));
        res.set_content(R"({"ok":true})", "application/json");
    });

    svr.listen("0.0.0.0", 8080);
    return 0;
}


#include <httplib.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <chrono>

// ---- 超簡易イベント発行（グローバル） ----
static std::mutex g_m;
static std::condition_variable g_cv;
static std::string g_msg;           // 次に送るSSEペイロード（丸ごと）
static std::atomic<bool> g_has_msg{false};
static std::atomic<bool> g_running{false};
static std::atomic<bool> g_cancel{false};
static std::atomic<long long> g_id{0};

static std::string sse(const char* event_name, const std::string& data_json) {
    long long id = ++g_id;
    // SSEは「空行\n\n」で区切る
    std::string out;
    out += "event: ";
    out += event_name;
    out += "\n";
    out += "id: " + std::to_string(id) + "\n";
    out += "data: " + data_json + "\n\n";
    return out;
}

static void publish(const std::string& payload) {
    {
        std::lock_guard<std::mutex> lk(g_m);
        g_msg = payload;
        g_has_msg.store(true);
    }
    g_cv.notify_all();
}

int main() {
    httplib::Server svr;

    // ---- SSE: クライアントはここに接続して待つ ----
    svr.Get("/events", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Content-Type", "text/event-stream; charset=utf-8");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        // 逆プロキシ(Nginx)を挟むならバッファ抑止が効くことがある
        // res.set_header("X-Accel-Buffering", "no");

        // 初回通知
        publish(sse("hello", R"({"msg":"connected"})"));

        res.set_chunked_content_provider(
            "text/event-stream",
            [](size_t, httplib::DataSink& sink) {
                while (true) {
                    std::string out;

                    // メッセージ待ち（15秒でタイムアウト→ping送る）
                    {
                        std::unique_lock<std::mutex> lk(g_m);
                        g_cv.wait_for(lk, std::chrono::seconds(15), [] {
                            return g_has_msg.load();
                        });

                        if (g_has_msg.load()) {
                            out = g_msg;
                            g_has_msg.store(false);
                        }
                    }

                    if (!out.empty()) {
                        if (!sink.write(out.data(), out.size())) break; // 切断
                    } else {
                        // keep-alive ping（コメント行）
                        const char* ping = ": ping\n\n";
                        if (!sink.write(ping, strlen(ping))) break;
                    }
                }
                sink.done();
                return true;
            }
        );
    });

    // ---- start: 例として擬似処理を開始（進捗→SSEで配信） ----
    svr.Post("/start", [](const httplib::Request&, httplib::Response& res) {
        if (g_running.exchange(true)) {
            res.set_content(R"({"ok":false,"reason":"already running"})", "application/json");
            return;
        }

        g_cancel.store(false);
        publish(sse("status", R"({"state":"started"})"));

        std::thread([] {
            for (int p = 0; p <= 100; p += 10) {
                if (g_cancel.load()) {
                    publish(sse("status", R"({"state":"canceled"})"));
                    g_running.store(false);
                    return;
                }
                publish(sse("progress", std::string(R"({"pct":)") + std::to_string(p) + "}"));
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }

            // 完了通知（結果本体は別APIにするのが堅牢だが、ここでは簡単にSSEで通知）
            publish(sse("complete", R"({"result":"OK"})"));
            g_running.store(false);
        }).detach();

        res.set_content(R"({"ok":true})", "application/json");
    });

    // ---- reset: 実行中なら止めて、状態をリセットしたことをSSEで通知 ----
    svr.Post("/reset", [](const httplib::Request&, httplib::Response& res) {
        g_cancel.store(true);
        publish(sse("reset", R"({"msg":"reset requested"})"));
        res.set_content(R"({"ok":true})", "application/json");
    });

    svr.listen("0.0.0.0", 8080);
}