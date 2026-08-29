#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "imgui.h"
#include "imgui_internal.h"

#include "tools/cabana/ui/widgets/cameraview.h"
#include "tools/cabana/ui/tools/routeinfo.h"
#include "tools/replay/logreader.h"
#include "tools/cabana/streams/replaystream.h"
#include "tools/cabana/ui/icons.h"

// bootstrap glyphs (utils::icon("name") in the Qt frontend)

// ToolButton (utils/qtutil.h): auto-raise icon button with a tooltip
bool toolButton(const char *icon, const char *tooltip, const char *id, float width = 0.0f);

// TabBar (utils/qtutil.h): QTabBar whose tabs get an "x" close button
class TabBar {
public:
  TabBar() = default;
  int addTab(const std::string &text);
  int count() const { return (int)tabs_.size(); }
  void setTabText(int index, const std::string &text) { tabs_[index].text = text; }
  void setTabData(int index, int data) { tabs_[index].data = data; }
  int tabData(int index) const { return index >= 0 && index < count() ? tabs_[index].data : 0; }  // QVariant().toInt() == 0
  int currentIndex() const { return current_index_; }
  void removeTab(int index);
  void setAutoHide(bool hide) { auto_hide_ = hide; }
  void setExpanding(bool) {}  // imgui tabs never expand
  void setTabsClosable(bool closable) { tabs_closable_ = closable; }  // QTabBar::setTabsClosable, off by default
  void draw();

  Observable<int> currentChanged;
  Observable<int> tabCloseRequested;

private:
  void closeTabClicked(int index) { tabCloseRequested(index); }
  struct Tab { std::string text; int data = 0; int id = 0; };
  std::vector<Tab> tabs_;
  int current_index_ = -1;
  int next_id_ = 0;
  bool select_current_ = false;  // programmatic current change, applied at the next draw()
  bool auto_hide_ = false;
  bool tabs_closable_ = false;
};

class Slider {
public:
  Slider();
  double currentSecond() const { return value() / factor; }
  void setCurrentSecond(double sec) { setValue(sec * factor); }
  void setTimeRange(double min, double max) { setRange(min * factor, max * factor); }
  // QSlider
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
  bool mouseLeft() const { return left_; }  // QEvent::Leave happened in the last draw()
  void draw();  // paintEvent + mousePressEvent
  const double factor = 1000.0;
  double thumbnail_dispaly_time = -1;

  Observable<> sliderReleased;

private:
  void mousePressEvent();
  void paintEvent();
  ImRect handleRect() const;  // QStyle::SC_SliderHandle
  int pixelPosToRangeValue(float x) const;  // QSliderPrivate::pixelPosToRangeValue
  int minimum_ = 0;
  int maximum_ = 99;
  int value_ = 0;
  bool slider_down_ = false;
  float click_offset_ = 0;  // QSliderPrivate::clickOffset: where inside the handle the drag started
  bool hovered_ = false;
  bool left_ = false;
  ImRect rect_;
};

class StreamCameraView : public CameraWidget {
public:
  StreamCameraView(std::string stream_name, VisionStreamType stream_type);
  void draw(const ImVec2 &size);  // paintEvent
  void parseQLog(std::shared_ptr<LogReader> qlog);

private:
  struct Thumbnail {
    RgbImage image;  // scaled; the border and alert are drawn at paint time
    std::optional<Timeline::Entry> alert;
  };
  Thumbnail generateThumbnail(const RgbImage &thumbnail, double seconds);
  void drawAlert(ImDrawList *p, const ImRect &rect, const Timeline::Entry &alert, float font_size);
  void drawThumbnail(ImDrawList *p);
  void drawScrubThumbnail(ImDrawList *p);
  void drawTime(ImDrawList *p, const ImRect &rect, double seconds);

  std::map<uint64_t, RgbImage> big_thumbnails;
  std::map<uint64_t, Thumbnail> thumbnails;
  GlTexture big_thumbnail_texture;  // the currently shown big thumbnail
  GlTexture thumbnail_texture;      // the currently shown thumbnail
  double thumbnail_dispaly_time = -1;
  friend class VideoWidget;
};

class VideoWidget {
public:
  VideoWidget();
  void draw();  // content only; MainWindow puts it in a child region above the charts
  // QWidget::setVisible of the video dock: MainWindow calls this every frame with the dock visibility so the
  // camera widget gets its showEvent/hideEvent (vipc thread start/stop) like it did in Qt
  void setVisible(bool visible);
  void showThumbnail(double seconds);
  std::string whatsThis() const { return whats_this_; }

protected:
  void eventFilter();  // MouseMove / Leave on the slider
  std::string formatTime(double sec, bool include_milliseconds = false);
  void timeRangeChanged();
  void updateState();
  void updatePlayBtnState();
  void createCameraWidget();
  void drawCameraWidget();
  void createPlaybackController();
  void drawPlaybackController();
  void skipToEnd();
  void toggleTimeDisplay();
  void createSpeedDropdown();
  void drawSpeedDropdown();
  void drawSpeedMenuItems();
  void loopPlaybackClicked();
  void vipcAvailableStreamsUpdated(std::set<VisionStreamType> streams);
  void showRouteInfo();

  std::unique_ptr<StreamCameraView> cam_widget;
  // time_display_action
  std::string time_text_;
  std::string time_tooltip_;
  // play_toggle_action
  const char *play_icon_ = icon::PLAY;
  std::string play_tooltip_;
  // speed_btn
  std::string speed_text_;
  int speed_index_ = -1;  // checked action of the speed menu
  // skip_to_end_action
  bool skip_to_end_enabled_ = true;
  // loop playback action
  const char *loop_icon_ = icon::REPEAT;
  std::unique_ptr<Slider> slider;
  std::unique_ptr<TabBar> camera_tab;
  std::vector<std::unique_ptr<RouteInfoDlg>> route_info_dlgs_;
  std::string whats_this_;
  Connections connections_;  // last: disconnected before the widgets its handlers dereference are destroyed
};
