# TODO: add install pip3 install rerun-sdk
import sys
import json
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.route import Route
import rerun as rr
import rerun.blueprint as rrb

is_first_run = True
topics = ['accelerometer', 'androidLog', 'cameraOdometry', 'can', 'carControl', 'carParams', 'carState', 'clocks', 'controlsState', 'deviceState',
          'driverCameraState', 'driverEncodeIdx', 'driverMonitoringState', 'driverStateV2', 'errorLogMessage', 'gnssMeasurements', 'gpsLocationExternal',
          'gyroscope', 'initData', 'lateralPlanDEPRECATED', 'lightSensor', 'liveCalibration', 'liveLocationKalman', 'liveParameters', 'liveTorqueParameters',
          'liveTracks', 'logMessage', 'longitudinalPlan', 'magnetometer', 'managerState', 'mapRenderState', 'microphone', 'modelV2', 'navInstruction',
          'navModel', 'navThumbnail', 'onroadEvents', 'pandaStates', 'peripheralState', 'procLog', 'qRoadEncodeIdx', 'radarState', 'roadCameraState',
          'roadEncodeIdx', 'sendcan', 'sentinel', 'temperatureSensor', 'thumbnail', 'ubloxGnss', 'ubloxRaw', 'uiDebug', 'uiPlan', 'uploaderState',
          'wideRoadCameraState', 'wideRoadEncodeIdx']


def convertBytesToString(data):
  if isinstance(data, bytes):
    return data.decode('latin-1')
  elif isinstance(data, list):
    return [convertBytesToString(item) for item in data]
  elif isinstance(data, dict):
    return {key: convertBytesToString(value) for key, value in data.items()}
  else:
    return data

# Log data to rerun, here could be values be excluded form plotting
def log_data(key, value):
      rr.log(str(key), rr.Scalar(value))
      ## Example of excluding a value from plotting
      # if key.find("timestamp") == -1:
      #   rr.log(str(key), rr.Scalar(value))


# Log every json attribute
def log_nested(obj, parent_key=''):
    for k, v in obj.items():
        if isinstance(v, dict):
            log_nested(v, parent_key + k + "/")
        elif isinstance(v, (int, float)):
            log_data(parent_key + k, v)
        elif isinstance(v, list):
            for index, item in enumerate(v):
                if isinstance(item, (int, float)):
                    log_data("/" + parent_key + k + "/" + str(index), item)
        # else:
        #     print("Not a plottable value")


# Create a blueprint for selected topics
def createBlueprint(topicsToGraph):
  print("Generating blueprint")
  timeSeriesViews = []
  for topic in topicsToGraph:
      timeSeriesViews.append(rrb.TimeSeriesView(name=topic, origin=f"/{topic}/"))
      rr.log(topic, rr.SeriesLine(name=topic), timeless=True)
  blueprint = rrb.Blueprint(
                          rrb.Vertical(
                            *timeSeriesViews,
                            rrb.SelectionPanel(expanded=False),
                            rrb.TimePanel(expanded=False)))
  return blueprint


def addGraphs(logPaths, topicsToGraph):
  print(f"Downloading logs [{len(logPaths)}]")
  counter = 1
  for logPath in logPaths:
    print(counter)
    counter+=1
    rlog = LogReader(logPath)
    for msg in rlog:
      global is_first_run
      if is_first_run:
        blueprint = createBlueprint(topicsToGraph)
        rr.init("rerun_test",spawn=True, default_blueprint=blueprint)
        is_first_run = False
      if msg.which() in topicsToGraph:
        jsonMsg = json.loads(json.dumps(convertBytesToString(msg.to_dict())))
        rr.set_time_sequence(msg.which(), msg.logMonoTime)
        log_nested(jsonMsg)


if __name__ == '__main__':
  if len(sys.argv) == 1:
    route_name = "a2a0ccea32023010|2023-07-27--13-01-19"
    print("No route was provided, using demo route")
  else:
    route_name = sys.argv[1]
  # Print menu
  print(f"What topics would you like to graph? [0-{len(topics)-1}]")
  for index, topic in enumerate(topics):
    print(str(index) + ". " + topic)
  # Get user input
  topicIndexes = [int(x) for x in (input("Separate your inputs with a space: ")).split(" ")]
  topicsToGraph = [topics[i] for i in topicIndexes]

  # Get logs for a route
  print("Getting route log paths")
  route = Route(route_name)
  logPaths = route.log_paths()
  addGraphs(logPaths, topicsToGraph)
