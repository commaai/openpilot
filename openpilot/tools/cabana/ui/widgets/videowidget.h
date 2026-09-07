#pragma once

#include <algorithm>
#include <future>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "imgui.h"
#include "imgui_internal.h"

#include "tools/cabana/ui/widgets/cameraview.h"
#include "tools/cabana/ui/widgets/tabbar.h"
#include "tools/cabana/ui/tools/routeinfo.h"
#include "tools/replay/logreader.h"
#include "tools/cabana/streams/replaystream.h"
#include "tools/cabana/ui/icons.h"

class Slider {
public:
  Slider() = default;
  double currentSecond() const { return value() / factor; }
  void setCurrentSecond(double sec) { setValue(sec * factor); }
  void setTimeRange(double min, double max) { setRange(min * factor, max * factor); }
  int value() const { return value_; }
  void setValue(int v) { value_ = std::clamp(v, minimum_, maximum_); }
  void setRange(int min, int max) { minimum_ = min; maximum_ = std::max(min, max); setValue(value_); }
  int minimum() const { return minimum_; }
  int maximum() const { return maximum_; }
  bool isSliderDown() const { return slider_down_; }
  float width() const { return rect_.GetWidth(); }
  const ImRect &rect() const { return rect_; }
  bool underMouse() const { return hovered_; }
  bool mouseLeft() const { return left_; }  // the mouse left the slider in the last draw()
  void draw(double thumbnail_time);  // thumbnail_time < 0: no thumbnail marker
  static constexpr double factor = 1000.0;

  Observable<> sliderReleased;

private:
  void handleMousePress();
  void paint(double thumbnail_time);
  ImRect handleRect() const;
  int pixelPosToRangeValue(float x) const;
  int minimum_ = 0;
  int maximum_ = 99;
  int value_ = 0;
  bool slider_down_ = false;
  float click_offset_ = 0;  // where inside the handle the drag started
  bool hovered_ = false;
  bool left_ = false;
  ImRect rect_;
};

class StreamCameraView : public CameraWidget {
public:
  StreamCameraView(std::string stream_name, VisionStreamType stream_type);
  ~StreamCameraView();
  void draw(const ImVec2 &size, double thumbnail_time);  // thumbnail_time < 0: no thumbnail
  void parseQLog(std::shared_ptr<LogReader> qlog);  // decodes the thumbnails on the thread pool

private:
  struct PendingThumbnails {
    std::future<void> done;
    std::shared_ptr<std::map<uint64_t, RgbImage>> thumbnails;
  };
  void collectThumbnails();  // moves the decoded thumbnails in once a parseQLog task is done
  // the first thumbnail at or after sec, uploaded to big_thumbnail_texture_; nullptr when there is none
  const RgbImage *thumbnailAt(double sec);
  void drawAlert(ImDrawList *p, const ImRect &rect, const Timeline::Entry &alert, float font_size);
  void drawThumbnail(ImDrawList *p, double sec);
  void drawScrubThumbnail(ImDrawList *p, double sec);
  void drawTime(ImDrawList *p, const ImRect &rect, double seconds);

  std::map<uint64_t, RgbImage> big_thumbnails_;
  GlTexture big_thumbnail_texture_;  // the currently shown thumbnail
  std::vector<PendingThumbnails> pending_thumbnails_;
};

class VideoWidget {
public:
  VideoWidget();
  void draw();  // content only; MainWindow puts it in a child region above the charts
  float sizeHintHeight() const;
  float defaultHeight(float width) const;
  // MainWindow calls this every frame with the video dock visibility, so the camera widget gets its
  // vipc thread started and stopped
  void setVisible(bool visible);
  void showThumbnail(double seconds);
  std::string whatsThis() const;

private:
  void updateSliderThumbnail();  // the thumbnail follows the mouse over the slider
  std::string formatTime(double sec, bool include_milliseconds = false);
  void timeRangeChanged();
  void createCameraWidget();
  void drawCameraWidget();
  void drawPlaybackController();
  void skipToEnd();
  void toggleTimeDisplay();
  void createSpeedDropdown();
  void drawSpeedMenuItems();
  void loopPlaybackClicked();
  void cropVideoClicked();
  void vipcAvailableStreamsUpdated(std::set<VisionStreamType> streams);
  void showRouteInfo();

  std::unique_ptr<StreamCameraView> cam_widget_;
  std::string speed_text_;
  int speed_index_ = -1;  // checked entry of the speed menu
  bool skip_to_end_enabled_ = true;
  bool msgs_received_ = false;  // the time is blank until the live stream delivers its first messages
  double thumbnail_display_time_ = -1;
  std::unique_ptr<Slider> slider_;
  std::unique_ptr<TabBar> camera_tab_;
  std::vector<std::unique_ptr<RouteInfoDlg>> route_info_dlgs_;
  Connections connections_;  // last: disconnected before the widgets its handlers dereference are destroyed
};
