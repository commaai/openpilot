#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tools/cabana/streams/abstractstream.h"
#include "tools/cabana/streams/pandastream.h"
#ifdef __linux__
#include "tools/cabana/streams/socketcanstream.h"
#endif
#include "tools/cabana/ui/dialogs/routesdialog.h"
#include "tools/cabana/ui/util.h"

class AbstractOpenStreamWidget {
public:
  virtual ~AbstractOpenStreamWidget() = default;
  virtual const char *title() const = 0;
  virtual void draw() = 0;
  // nested dialogs, drawn at the modal's level: a popup opened from inside the tab's child window is
  // never the top-most modal's own call site, so PopupOwner would skip it
  virtual void drawPopups() {}
  virtual std::unique_ptr<AbstractStream> open() = 0;
  virtual bool openEnabled() const { return true; }
};

class OpenReplayWidget : public AbstractOpenStreamWidget {
public:
  const char *title() const override { return "Replay"; }
  void draw() override;
  void drawPopups() override;
  std::unique_ptr<AbstractStream> open() override;

private:
  std::string route_;
  bool cameras_[3] = {true, false, false};
  RoutesDialog routes_dialog_;
  // guards dialog continuations that outlive the stream selector
  std::shared_ptr<bool> alive_ = std::make_shared<bool>(true);
};

class OpenPandaWidget : public AbstractOpenStreamWidget {
public:
  OpenPandaWidget();
  const char *title() const override { return "Panda"; }
  void draw() override;
  std::unique_ptr<AbstractStream> open() override;
  bool openEnabled() const override { return !already_connected_; }

private:
  void refreshSerials();
  void buildConfigForm();

  bool already_connected_ = false;
  std::vector<std::string> serials_;
  int serial_index_ = 0;
  bool has_panda_ = false;
  bool has_fd_ = false;
  std::vector<int> can_speed_index_, data_speed_index_;
  PandaStreamConfig config = {};
};

class OpenDeviceWidget : public AbstractOpenStreamWidget {
public:
  const char *title() const override { return "Device"; }
  void draw() override;
  std::unique_ptr<AbstractStream> open() override;

private:
  int mode_ = 1;  // 0 = MSGQ, 1 = ZMQ
  std::string ip_address_;
};

#ifdef __linux__
class OpenSocketCanWidget : public AbstractOpenStreamWidget {
public:
  OpenSocketCanWidget();
  const char *title() const override { return "SocketCAN"; }
  void draw() override;
  std::unique_ptr<AbstractStream> open() override;

private:
  void refreshDevices();

  std::vector<std::string> devices_;
  int device_index_ = 0;
  SocketCanStreamConfig config = {};
};
#endif

class StreamSelector {
public:
  using Callback = std::function<void(std::unique_ptr<AbstractStream> stream, const std::string &dbc_file)>;
  // on_done gets a null stream on cancel
  void open(Callback on_done);
  void draw();

private:
  bool open_ = false;
  PopupOwner popup_;
  bool first_frame_ = false;
  std::string dbc_file_;
  std::vector<std::unique_ptr<AbstractOpenStreamWidget>> widgets_;
  Callback on_done_;
};
