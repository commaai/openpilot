// video in the browser: the replay threads demux files into Annex B access units, the page decodes them
// with WebCodecs and uploads straight into GL textures that the camera widget draws.
#include <emscripten.h>
#include <emscripten/threading.h>

#include <cstring>
#include <string>
#include <vector>

#include "openpilot/cereal/gen/cpp/log.capnp.h"
#include "common/util.h"
#include "tools/cabana/web/web.h"
#include "tools/replay/camera.h"
#include "tools/replay/framereader.h"
#include "tools/replay/py_tools.h"

namespace {

constexpr const char *CODEC_H264 = "avc1.64001f";
constexpr const char *CODEC_HEVC = "hev1.1.6.L120.B0";
constexpr int MAX_SEQUENTIAL_GAP = 30;  // decode forward through at most this many skipped frames

bool isKeyNal(const uint8_t *nal, bool hevc) {
  int type = hevc ? (nal[0] >> 1) & 0x3f : nal[0] & 0x1f;
  return hevc ? (type >= 16 && type <= 21) : type == 5;
}

// offsets of NAL payloads (after the start code) in an Annex B buffer
std::vector<size_t> nalUnits(const uint8_t *data, size_t size) {
  std::vector<size_t> nals;
  for (size_t i = 0; i + 3 < size; ++i) {
    if (data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1) {
      nals.push_back(i + 3);
      i += 2;
    }
  }
  return nals;
}

void addAccessUnit(FrameReader &fr, const uint8_t *au, size_t size, bool hevc) {
  if (size == 0) return;
  bool key = false;
  for (size_t nal : nalUnits(au, size)) {
    if (nal < size && isKeyNal(au + nal, hevc)) key = true;
  }
  fr.packets.push_back({fr.data.size(), size, key});
  fr.data.insert(fr.data.end(), au, au + size);
}

// MPEG-TS with one H.264 or HEVC program, as qcamera.ts
bool demuxTs(FrameReader &fr, const std::string &ts) {
  const auto *d = (const uint8_t *)ts.data();
  int pmt_pid = -1, video_pid = -1;
  bool hevc = false, in_pes = false;
  std::vector<uint8_t> pes;
  for (size_t i = 0; i + 188 <= ts.size(); i += 188) {
    if (d[i] != 0x47) continue;
    const bool pusi = d[i + 1] & 0x40;
    const int pid = ((d[i + 1] & 0x1f) << 8) | d[i + 2];
    const int afc = (d[i + 3] >> 4) & 3;
    size_t p = i + 4;
    if (afc & 2) p += 1 + d[p];
    if (!(afc & 1) || p >= i + 188) continue;
    const size_t end = i + 188;
    if (pid == 0 && pmt_pid < 0) {
      size_t q = p + 1 + d[p];
      int section_length = ((d[q + 1] & 0x0f) << 8) | d[q + 2];
      for (size_t r = q + 8; r + 4 <= q + 3 + section_length - 4 && r + 4 <= end; r += 4) {
        int program = (d[r] << 8) | d[r + 1];
        if (program != 0) pmt_pid = ((d[r + 2] & 0x1f) << 8) | d[r + 3];
      }
    } else if (pid == pmt_pid && video_pid < 0) {
      size_t q = p + 1 + d[p];
      int section_length = ((d[q + 1] & 0x0f) << 8) | d[q + 2];
      int info_length = ((d[q + 10] & 0x0f) << 8) | d[q + 11];
      for (size_t r = q + 12 + info_length; r + 5 <= q + 3 + section_length - 4 && r + 5 <= end;) {
        int stream_type = d[r];
        int es_pid = ((d[r + 1] & 0x1f) << 8) | d[r + 2];
        int es_length = ((d[r + 3] & 0x0f) << 8) | d[r + 4];
        if (stream_type == 0x1b || stream_type == 0x24) {
          video_pid = es_pid;
          hevc = stream_type == 0x24;
        }
        r += 5 + es_length;
      }
    } else if (pid == video_pid) {
      if (pusi) {
        addAccessUnit(fr, pes.data(), pes.size(), hevc);
        pes.clear();
        if (end - p >= 9 && d[p] == 0 && d[p + 1] == 0 && d[p + 2] == 1) {
          p += 9 + d[p + 8];  // PES header
        }
        in_pes = true;
      }
      if (in_pes && p < end) pes.insert(pes.end(), d + p, d + end);
    }
  }
  addAccessUnit(fr, pes.data(), pes.size(), hevc);
  fr.codec = hevc ? CODEC_HEVC : CODEC_H264;
  return video_pid >= 0 && !fr.packets.empty();
}

// raw HEVC Annex B, as the device's fcamera/ecamera/dcamera files
bool splitHevc(FrameReader &fr, const std::string &stream) {
  const auto *d = (const uint8_t *)stream.data();
  std::vector<size_t> nals = nalUnits(d, stream.size());
  size_t au_start = 0;
  bool has_slice = false;
  for (size_t n = 0; n < nals.size(); ++n) {
    size_t nal = nals[n];
    size_t nal_start = nal - 3 - (nal >= 4 && d[nal - 4] == 0 ? 1 : 0);
    int type = (d[nal] >> 1) & 0x3f;
    bool slice = type < 32;
    bool first_slice = slice && (d[nal + 2] & 0x80);
    bool starts_au = has_slice && ((type >= 32 && type <= 35) || first_slice);
    if (starts_au) {
      addAccessUnit(fr, d + au_start, nal_start - au_start, true);
      au_start = nal_start;
      has_slice = false;
    }
    if (slice) has_slice = true;
  }
  addAccessUnit(fr, d + au_start, stream.size() - au_start, true);
  fr.codec = CODEC_HEVC;
  return !fr.packets.empty();
}

// [u32 size][u8 key][bytes] per access unit, the last one is the frame to show
std::vector<uint8_t> packRange(const FrameReader &fr, int from, int to) {
  std::vector<uint8_t> out;
  for (int i = from; i <= to; ++i) {
    const auto &pkt = fr.packets[i];
    uint32_t size = pkt.size;
    out.insert(out.end(), (uint8_t *)&size, (uint8_t *)&size + 4);
    out.push_back(pkt.key);
    out.insert(out.end(), fr.data.begin() + pkt.offset, fr.data.begin() + pkt.offset + pkt.size);
  }
  return out;
}

web::video::Frame latest_frames[MAX_CAMERAS];

}  // namespace

// clang-format off
EM_JS(void, webVideoSubmit, (int cam, const char *codec_ptr, uint8_t *ptr, int len, int reset), {
  const codec = UTF8ToString(codec_ptr);
  const bytes = HEAPU8.slice(ptr, ptr + len);
  _free(ptr);
  if (!Module.video) Module.video = {};
  let d = Module.video[cam];
  if (!d) {
    d = Module.video[cam] = { decoder: null, codec: null, texture: 0, timestamp: 0, targets: new Set() };
  }
  const output = (frame) => {
    if (d.targets.delete(frame.timestamp)) {
      const gl = GL.currentContext.GLctx;
      if (!d.texture) {
        const tex = gl.createTexture();
        d.texture = GL.getNewId(GL.textures);
        tex.name = d.texture;
        GL.textures[d.texture] = tex;
      }
      gl.bindTexture(gl.TEXTURE_2D, GL.textures[d.texture]);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, frame);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      _web_video_frame(cam, d.texture, frame.displayWidth, frame.displayHeight);
    }
    frame.close();
  };
  const configure = () => {
    if (d.decoder && d.decoder.state !== "closed") d.decoder.close();
    d.decoder = new VideoDecoder({ output, error: (e) => console.error("VideoDecoder", codec, e.message) });
    d.decoder.configure({ codec, optimizeForLatency: true });
    d.codec = codec;
    d.targets.clear();
  };
  if (!d.decoder || d.decoder.state === "closed" || d.codec !== codec || reset) configure();

  const view = new DataView(bytes.buffer);
  let off = 0;
  const chunks = [];
  while (off + 5 <= bytes.length) {
    const size = view.getUint32(off, true);
    const key = bytes[off + 4] !== 0;
    chunks.push({ key, data: bytes.subarray(off + 5, off + 5 + size) });
    off += 5 + size;
  }
  chunks.forEach((c, i) => {
    const timestamp = d.timestamp++;
    if (i === chunks.length - 1) d.targets.add(timestamp);
    d.decoder.decode(new EncodedVideoChunk({ type: c.key ? "key" : "delta", timestamp, data: c.data }));
  });
});
// clang-format on

extern "C" EMSCRIPTEN_KEEPALIVE void web_video_frame(int cam, unsigned int texture, int width, int height) {
  if (cam < 0 || cam >= MAX_CAMERAS) return;
  latest_frames[cam] = {texture, width, height};
}

namespace web::video {
Frame latest(CameraType type) {
  return latest_frames[type];
}
}  // namespace web::video

bool FrameReader::load(CameraType type, const std::string &url, bool no_hw_decoder, std::atomic<bool> *abort, bool local_cache) {
  const std::string path = PyTools::download(url, local_cache, abort);
  if (path.empty()) return false;
  const std::string file = util::read_file(path);
  if (file.empty()) return false;
  const bool ok = util::ends_with(getUrlWithoutQuery(url), ".ts") ? demuxTs(*this, file) : splitHevc(*this, file);
  if (!ok) logMessage(ReplyMsgType::Warning, "no video frames in %s", getUrlWithoutQuery(url).c_str());
  return ok;
}

void CameraServer::pushFrame(CameraType type, FrameReader *fr, const Event *event) {
  capnp::FlatArrayMessageReader reader(event->data);
  auto evt = reader.getRoot<cereal::Event>();
  auto eidx = capnp::AnyStruct::Reader(evt).getPointerSection()[0].getAs<cereal::EncodeIndex>();
  const int idx = eidx.getSegmentId();
  if (idx < 0 || idx >= (int)fr->packets.size()) return;

  int from = idx;
  bool reset = true;
  if (fr == last_reader_[type] && idx > last_idx_[type] && idx - last_idx_[type] <= MAX_SEQUENTIAL_GAP) {
    from = last_idx_[type] + 1;
    reset = false;
  } else {
    while (from > 0 && !fr->packets[from].key) --from;
  }
  last_reader_[type] = fr;
  last_idx_[type] = idx;

  std::vector<uint8_t> packed = packRange(*fr, from, idx);
  auto *buf = (uint8_t *)malloc(packed.size());
  memcpy(buf, packed.data(), packed.size());
  // the page owns the decoders; the buffer is freed there
  MAIN_THREAD_ASYNC_EM_ASM({ webVideoSubmit($0, $1, $2, $3, $4); }, (int)type, fr->codec, buf, (int)packed.size(), (int)reset);
}
