#include <iostream>
#include <selfdrive/ui/ui.h>

#include "application.h"
#include "tools/replay/replay.h"

void startReplayThread() {
  std::vector<std::string> allow = (std::vector<std::string>{
    "modelV2", "controlsState", "liveCalibration", "radarState", "deviceState",
    "pandaStates", "carParams", "driverMonitoringState", "carState", "driverStateV2",
    "wideRoadCameraState", "managerState", "selfdriveState", "longitudinalPlan",
  });

  std::vector<std::string> block;
  Replay replay("a2a0ccea32023010|2023-07-27--13-01-19", allow, block);

  if (!replay.load()) {
    return;
  }

  std::cout << "Replay started." << std::endl;
  replay.setEndSeconds(66);
  replay.start(60);
  replay.waitUntilEnd();
  std::cout << "Replay ended." << std::endl;
  raise(SIGINT);
}

int main(int argc, char *argv[]) {
  Application a(argc, argv);

  std::thread thread(startReplayThread);
  std::this_thread::sleep_for(std::chrono::milliseconds(2000));

  if (a.exec()) {
    std::cerr << "Failed to start app." << std::endl;
  }

  thread.join();
  a.close();
  return 0;
}