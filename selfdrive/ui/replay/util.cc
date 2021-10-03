#include "selfdrive/ui/replay/util.h"

#include <cassert>
#include <bzlib.h>
#include <curl/curl.h>

#include "selfdrive/common/util.h"

struct CURLGlobalInitializer {
  CURLGlobalInitializer() { curl_global_init(CURL_GLOBAL_DEFAULT); }
  ~CURLGlobalInitializer() { curl_global_cleanup(); }
};

struct MultiPartWriter {
  int64_t offset;
  int64_t end;
  FILE *fp;
};

static size_t write_cb(char *data, size_t n, size_t l, void *userp) {
  MultiPartWriter *w = (MultiPartWriter *)userp;
  fseek(w->fp, w->offset, SEEK_SET);
  fwrite(data, l, n, w->fp);
  w->offset += n * l;
  return n * l;
}

static size_t dumy_write_cb(char *data, size_t n, size_t l, void *userp) { return n * l; }

int64_t getDownloadContentLength(const std::string &url) {
  CURL *curl = curl_easy_init();
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, dumy_write_cb);
  curl_easy_setopt(curl, CURLOPT_HEADER, 1);
  curl_easy_setopt(curl, CURLOPT_NOBODY, 1);
  CURLcode res = curl_easy_perform(curl);
  double content_length = -1;
  if (res == CURLE_OK) {
    res = curl_easy_getinfo(curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD, &content_length);
  }
  curl_easy_cleanup(curl);
  return res == CURLE_OK ? (int64_t)content_length : -1;
}

bool httpMultiPartDownload(const std::string &url, const std::string &target_file, int parts, std::atomic<bool> *abort) {
  static CURLGlobalInitializer curl_initializer;

  int64_t content_length = getDownloadContentLength(url);
  if (content_length == -1) return false;

  std::string tmp_file = target_file + ".tmp";
  FILE *fp = fopen(tmp_file.c_str(), "wb");
  // create a sparse file
  fseek(fp, content_length, SEEK_SET);

  CURLM *cm = curl_multi_init();
  std::map<CURL *, MultiPartWriter> writers;
  const int part_size = content_length / parts;
  for (int i = 0; i < parts; ++i) {
    CURL *eh = curl_easy_init();
    writers[eh] = {
        .fp = fp,
        .offset = i * part_size,
        .end = i == parts - 1 ? content_length - 1 : (i + 1) * part_size - 1,
    };
    curl_easy_setopt(eh, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(eh, CURLOPT_WRITEDATA, (void *)(&writers[eh]));
    curl_easy_setopt(eh, CURLOPT_URL, url.c_str());
    curl_easy_setopt(eh, CURLOPT_RANGE, util::string_format("%d-%d", writers[eh].offset, writers[eh].end).c_str());
    curl_easy_setopt(eh, CURLOPT_HTTPGET, 1);
    curl_easy_setopt(eh, CURLOPT_NOSIGNAL, 1);
    curl_easy_setopt(eh, CURLOPT_FOLLOWLOCATION, 1);
    curl_multi_add_handle(cm, eh);
  }

  int running = 1, success_cnt = 0;
  while (!(abort && abort->load())) {
    CURLMcode ret = curl_multi_perform(cm, &running);

    if (!running) {
      CURLMsg *msg;
      int msgs_left = -1;
      while ((msg = curl_multi_info_read(cm, &msgs_left))) {
        if (msg->msg == CURLMSG_DONE && msg->data.result == CURLE_OK) {
          int http_status_code = 0;
          curl_easy_getinfo(msg->easy_handle, CURLINFO_RESPONSE_CODE, &http_status_code);
          success_cnt += (http_status_code == 206);
        }
      }
      break;
    }

    if (ret == CURLM_OK) {
      curl_multi_wait(cm, nullptr, 0, 1000, nullptr);
    }
  };

  fclose(fp);
  bool success = success_cnt == parts;
  if (success) {
    success = ::rename(tmp_file.c_str(), target_file.c_str()) == 0;
  }

  // cleanup curl
  for (auto &[e, w] : writers) {
    curl_multi_remove_handle(cm, e);
    curl_easy_cleanup(e);
  }
  curl_multi_cleanup(cm);
  return success;
}

bool decompressBZ2(std::vector<uint8_t> &dest, const char srcData[], size_t srcSize, size_t outputSizeIncrement = 0x100000U) {
  bz_stream strm = {};
  int ret = BZ2_bzDecompressInit(&strm, 0, 0);
  assert(ret == BZ_OK);

  strm.next_in = const_cast<char *>(srcData);
  strm.avail_in = srcSize;
  do {
    strm.next_out = (char *)&dest[strm.total_out_lo32];
    strm.avail_out = dest.size() - strm.total_out_lo32;
    ret = BZ2_bzDecompress(&strm);
    if (ret == BZ_OK && strm.avail_in > 0 && strm.avail_out == 0) {
      dest.resize(dest.size() + outputSizeIncrement);
    }
  } while (ret == BZ_OK);

  BZ2_bzDecompressEnd(&strm);
  dest.resize(strm.total_out_lo32);
  return ret == BZ_STREAM_END;
}