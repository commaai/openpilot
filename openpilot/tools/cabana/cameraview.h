#pragma once

#include <atomic>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "msgq/visionipc/visionipc_client.h"
#include "tools/cabana/imguihost.h"

class CameraWidget : public ImGuiHost {
  Q_OBJECT

public:
  explicit CameraWidget(std::string stream_name, VisionStreamType stream_type, QWidget* parent = nullptr);
  ~CameraWidget();
  void setStreamType(VisionStreamType type) { requested_stream_type = type; }
  VisionStreamType getStreamType() { return active_stream_type; }
  void stopVipcThread();

signals:
  void clicked();
  void vipcThreadFrameReceived();
  void vipcAvailableStreamsUpdated(std::set<VisionStreamType>);

protected:
  void drawFrame() override;
  void showEvent(QShowEvent *event) override;
  void hideEvent(QHideEvent *event) override { stopVipcThread(); }
  void mouseReleaseEvent(QMouseEvent *event) override {
    ImGuiHost::mouseReleaseEvent(event);
    emit clicked();
  }
  void vipcThread();
  void clearFrames();

  // guarded by frame_lock: written by vipc thread, uploaded by GUI thread
  std::vector<uint8_t> rgb_frame;  // RGBA, tightly packed
  int frame_width = 0, frame_height = 0;
  uint64_t frame_gen = 0;

  std::vector<uint8_t> rgb_back;  // vipc thread only

  // GUI thread only
  uint32_t texture = 0;
  int tex_width = 0, tex_height = 0;
  bool tex_valid = false;
  uint64_t uploaded_gen = 0;

  std::string stream_name;
  std::atomic<VisionStreamType> active_stream_type;
  std::atomic<VisionStreamType> requested_stream_type;
  std::set<VisionStreamType> available_streams;
  std::thread vipc_thread;
  std::atomic<bool> vipc_exit = false;
  std::mutex frame_lock;

protected slots:
  void vipcFrameReceived();
  void availableStreamsUpdated(std::set<VisionStreamType> streams);
};

Q_DECLARE_METATYPE(std::set<VisionStreamType>);
