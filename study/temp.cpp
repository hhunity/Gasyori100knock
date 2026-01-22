#include "httplib.h"
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdio>

namespace fs = std::filesystem;

struct FileItem {
    std::string name; // UTF-8
    std::string path; // "upload/xxx"
    uintmax_t size;
};

// JSON文字列を最低限エスケープ（" と \ と制御文字）
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
        case '\"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20) {
                char buf[7];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            } else {
                out += static_cast<char>(c);
            }
        }
    }
    return out;
}

int main() {
    httplib::Server svr;

    const fs::path upload_dir = fs::absolute("upload"); // 固定ディレクトリ

    svr.Get("/api/uploads", [&](const httplib::Request&, httplib::Response& res) {
        if (!fs::exists(upload_dir) || !fs::is_directory(upload_dir)) {
            res.status = 404;
            res.set_content(R"({"error":"upload dir not found"})", "application/json; charset=utf-8");
            return;
        }

        std::vector<FileItem> items;
        items.reserve(256);

        for (const auto& entry : fs::directory_iterator(upload_dir)) {
            if (!entry.is_regular_file()) continue;

            const auto filename_u8 = entry.path().filename().u8string(); // UTF-8
            const auto rel = fs::path("upload") / entry.path().filename();
            const auto rel_u8 = rel.generic_u8string(); // "upload/xxx"

            FileItem it;
            it.name = filename_u8;
            it.path = rel_u8;
            it.size = entry.file_size();
            items.push_back(std::move(it));
        }

        // 必要ならソート（名前順）
        std::sort(items.begin(), items.end(),
                  [](const FileItem& a, const FileItem& b) { return a.name < b.name; });

        std::string body;
        body.reserve(256 + items.size() * 128);

        body += "{";
        body += "\"dir\":\"upload\",";
        body += "\"count\":" + std::to_string(items.size()) + ",";
        body += "\"files\":[";

        for (size_t i = 0; i < items.size(); ++i) {
            const auto& f = items[i];
            if (i > 0) body += ",";

            body += "{";
            body += "\"name\":\"" + json_escape(f.name) + "\",";
            body += "\"path\":\"" + json_escape(f.path) + "\",";
            body += "\"size\":" + std::to_string(f.size);
            body += "}";
        }

        body += "]}";

        res.set_header("Cache-Control", "no-store");
        res.set_content(body, "application/json; charset=utf-8");
    });

    // upload/ の中身をそのまま静的配信したい場合（任意）
    // GET /upload/a.tif で取れるようになる
    svr.set_mount_point("/upload", upload_dir.string().c_str());

    svr.listen("0.0.0.0", 8080);
}