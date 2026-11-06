// browser implementation of the helpers that shell out to python natively: the comma API and
// file downloads go through fetch, decompression is in-process, and downloads land in MEMFS.
// Every function here runs on a worker thread (fetch is synchronous), never on the main thread.
#include "tools/replay/py_tools.h"

#include <sys/stat.h>

#include <bzlib.h>
#include <emscripten/fetch.h>
#include <zstd/zstd.h>

#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>

#include "tools/replay/util.h"
#include "tools/cabana/web/web.h"

namespace {

constexpr const char *API_HOST = "https://api.commadotai.com/";
constexpr const char *CACHE_DIR = "/cache";

DownloadProgressHandler progress_handler = nullptr;
std::mutex token_mutex;
std::string api_token;

struct Response {
  int status = 0;
  std::string body;
};

Response httpGet(const std::string &url, const std::string &auth = "") {
  emscripten_fetch_attr_t attr;
  emscripten_fetch_attr_init(&attr);
  strcpy(attr.requestMethod, "GET");
  attr.attributes = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY | EMSCRIPTEN_FETCH_SYNCHRONOUS;
  const char *headers[] = {"Authorization", auth.c_str(), nullptr};
  if (!auth.empty()) attr.requestHeaders = headers;
  emscripten_fetch_t *fetch = emscripten_fetch(&attr, url.c_str());
  Response r;
  if (fetch) {
    r.status = fetch->status;
    if (fetch->numBytes > 0) r.body.assign(fetch->data, fetch->numBytes);
    emscripten_fetch_close(fetch);
  }
  return r;
}

std::string authHeader() {
  std::lock_guard lk(token_mutex);
  return api_token.empty() ? "" : "JWT " + api_token;
}

// mirrors file_downloader.api_call: the API's JSON, or {"error": ...}
std::string apiGet(const std::string &endpoint) {
  Response r = httpGet(API_HOST + endpoint, authHeader());
  if (r.status == 200) return r.body;
  const char *error = r.status == 401 || r.status == 403 ? "unauthorized" : r.status == 404 ? "not_found" : nullptr;
  if (error) return std::string("{\"error\": \"") + error + "\"}";
  return "{\"error\": \"HTTP " + std::to_string(r.status) + " from " + endpoint + "\"}";
}

bool decompressBz2(const std::string &in, std::string &out) {
  bz_stream s = {};
  if (BZ2_bzDecompressInit(&s, 0, 0) != BZ_OK) return false;
  s.next_in = const_cast<char *>(in.data());
  s.avail_in = in.size();
  out.resize(in.size() * 4);
  int ret = BZ_OK;
  size_t produced = 0;
  while (ret == BZ_OK) {
    if (produced == out.size()) out.resize(out.size() * 2);
    s.next_out = out.data() + produced;
    s.avail_out = out.size() - produced;
    ret = BZ2_bzDecompress(&s);
    produced = out.size() - s.avail_out;
  }
  BZ2_bzDecompressEnd(&s);
  out.resize(produced);
  return ret == BZ_STREAM_END;
}

bool decompressZstd(const std::string &in, std::string &out) {
  ZSTD_DStream *ds = ZSTD_createDStream();
  if (!ds) return false;
  ZSTD_initDStream(ds);
  ZSTD_inBuffer input = {in.data(), in.size(), 0};
  out.resize(in.size() * 4);
  size_t produced = 0, ret = 1;
  while (input.pos < input.size && !ZSTD_isError(ret)) {
    if (produced == out.size()) out.resize(out.size() * 2);
    ZSTD_outBuffer output = {out.data(), out.size(), produced};
    ret = ZSTD_decompressStream(ds, &output, &input);
    produced = output.pos;
  }
  ZSTD_freeDStream(ds);
  out.resize(produced);
  return !ZSTD_isError(ret) && ret == 0;
}

std::string cachePath(const std::string &url) {
  char name[32];
  snprintf(name, sizeof(name), "%016zx", std::hash<std::string>{}(getUrlWithoutQuery(url)));
  return std::string(CACHE_DIR) + "/" + name;
}

bool writeFile(const std::string &path, const std::string &data) {
  FILE *f = fopen(path.c_str(), "wb");
  if (!f) return false;
  bool ok = fwrite(data.data(), 1, data.size(), f) == data.size();
  fclose(f);
  return ok;
}

}  // namespace

namespace web {
void setApiToken(const std::string &token) {
  std::lock_guard lk(token_mutex);
  api_token = token;
}
bool hasApiToken() {
  std::lock_guard lk(token_mutex);
  return !api_token.empty();
}
}  // namespace web

void installDownloadProgressHandler(DownloadProgressHandler handler) {
  progress_handler = handler;
}

namespace PyTools {

std::string download(const std::string &url, bool use_cache, std::atomic<bool> *abort) {
  mkdir(CACHE_DIR, 0755);
  const std::string path = cachePath(url);
  struct stat st;
  if (use_cache && stat(path.c_str(), &st) == 0) return path;

  Response r = httpGet(url);
  if (r.status != 200 && r.status != 206) {
    logMessage(ReplyMsgType::Warning, "HTTP %d downloading %s", r.status, getUrlWithoutQuery(url).c_str());
    if (progress_handler) progress_handler(0, 0, false);
    return "";
  }
  if (abort && *abort) return "";

  std::string data;
  bool ok = true;
  if (r.body.compare(0, 3, "BZh") == 0) {
    ok = decompressBz2(r.body, data);
  } else if (r.body.compare(0, 4, "\x28\xb5\x2f\xfd") == 0) {
    ok = decompressZstd(r.body, data);
  } else {
    data.swap(r.body);
  }
  if (!ok) {
    logMessage(ReplyMsgType::Warning, "failed to decompress %s", getUrlWithoutQuery(url).c_str());
    if (progress_handler) progress_handler(0, 0, false);
    return "";
  }
  if (!writeFile(path, data)) return "";
  if (progress_handler) progress_handler(r.body.size(), r.body.size(), true);
  return path;
}

std::string decompress(const std::string &path, std::atomic<bool> *abort) {
  // only remote files reach the browser, and download() already decompressed those
  return "";
}

std::string getRouteFiles(const std::string &route) {
  return apiGet("v1/route/" + route + "/files");
}

std::string resolveRouteFiles(const std::string &route, int begin, int end, const std::string &selector) {
  return "{\"error\": \"auto source is not available in the browser\"}";
}

std::string authenticate(const std::string &provider, std::atomic<bool> *abort) {
  return "{\"error\": \"Sign in at https://jwt.comma.ai and open cabana with #jwt=<token> once; the token is then remembered by this browser.\"}";
}

std::string getDevices() {
  return apiGet("v1/me/devices/");
}

std::string getDeviceRoutes(const std::string &dongle_id, int64_t start_ms, int64_t end_ms, bool preserved) {
  if (preserved) return apiGet("v1/devices/" + dongle_id + "/routes/preserved");
  std::string query;
  if (start_ms) query += "start=" + std::to_string(start_ms);
  if (end_ms) query += (query.empty() ? "" : "&") + std::string("end=") + std::to_string(end_ms);
  return apiGet("v1/devices/" + dongle_id + "/routes_segments" + (query.empty() ? "" : "?" + query));
}

std::string migrateLog(const std::string &log_path) {
  return "";  // cabana only reads CAN, which was never migrated
}

}  // namespace PyTools
