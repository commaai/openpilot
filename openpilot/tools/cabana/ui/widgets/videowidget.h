#pragma once

#include <algorithm>
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
  Slider();
  double currentSecond() const { return value() / factor; }
  void setCurrentSecond(double sec) { setValue(sec * factor); }
  void setTimeRange(double min, double max) { setRange(min * factor, max * factor); }
  int value() const { return value_; }
  void setValue(int v) { value_ = std::clamp(v, minimum_, maximum_); }
  void setRange(int min, int max) { minimum_ = min; maximum_ = std::max(min, max); setValue(value_); }
  int minimum() const { return minimum_; }
  int maximum() const { return maximum_; }
  bool isSliderDown() const { return slider_down_; }
  void setSingleStep(int) {}
  float width() const { return rect_.GetWidth(); }
  const ImRect &rect() const { return rect_; }
  bool underMouse() const { return hovered_; }
  bool mouseLeft() const { return left_; }  // the mouse left the slider in the last draw()
  void draw();
  const double factor = 1000.0;
  double thumbnail_dispaly_time = -1;

  Observable<> sliderReleased;

private:
  void mousePressEvent();
  void paintEvent();
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
  void draw(const ImVec2 &size);
  void parseQLog(std::shared_ptr<LogReader> qlog);

private:
  void drawAlert(ImDrawList *p, const ImRect &rect, const Timeline::Entry &alert, float font_size);
  void drawThumbnail(ImDrawList *p);
  void drawScrubThumbnail(ImDrawList *p);
  void drawTime(ImDrawList *p, const ImRect &rect, double seconds);

  std::map<uint64_t, RgbImage> big_thumbnails;
  GlTexture big_thumbnail_texture;  // the currently shown thumbnail
  double thumbnail_dispaly_time = -1;
  friend class VideoWidget;
};

class VideoWidget {
public:
  VideoWidget();
  void draw();  // content only; MainWindow puts it in a child region above the charts
  float sizeHintHeight() const;
  float defaultHeight(float width) const;
  // MainWindow calls this every frame with the video dock visibility, so the camera widget gets its
  // showEvent/hideEvent (vipc thread start/stop)
  void setVisible(bool visible);
  void showThumbnail(double seconds);
  std::string whatsThis() const;

protected:
  void eventFilter();  // mouse move / leave on the slider
  std::string formatTime(double sec, bool include_milliseconds = false);
  void timeRangeChanged();
  void createCameraWidget();
  void drawCameraWidget();
  void drawPlaybackController();
  void skipToEnd();
  void toggleTimeDisplay();
  void createSpeedDropdown();
  void drawSpeedDropdown(float width);
  void drawSpeedMenuItems();
  void loopPlaybackClicked();
  void vipcAvailableStreamsUpdated(std::set<VisionStreamType> streams);
  void showRouteInfo();

  std::unique_ptr<StreamCameraView> cam_widget;
  std::string speed_text_;
  int speed_index_ = -1;  // checked entry of the speed menu
  bool skip_to_end_enabled_ = true;
  bool time_tooltip_shown_ = false;
  std::unique_ptr<Slider> slider;
  std::unique_ptr<TabBar> camera_tab;
  std::vector<std::unique_ptr<RouteInfoDlg>> route_info_dlgs_;
  Connections connections_;  // last: disconnected before the widgets its handlers dereference are destroyed
};
