#!/usr/bin/env python3
import sys
import json
import base64
import os
import subprocess
from multiprocessing import Pool
from openpilot.tools.lib.route import Route
from openpilot.tools.lib.logreader import LogReader

try:
  from mcap.writer import Writer, CompressionType
except ImportError:
  print("mcap module not found. Attempting to install...")
  subprocess.run([sys.executable, "-m", "pip", "install", "mcap"])
  # Attempt to import again after installation
  try:
    from mcap.writer import Writer, CompressionType
  except ImportError:
    print("Failed to install mcap module. Exiting.")
    sys.exit(1)


FOXGLOVE_IMAGE_SCHEME_TITLE = "foxglove.CompressedImage"
FOXGLOVE_GEOJSON_TITLE = "foxglove.GeoJSON"
FOXGLOVE_IMAGE_ENCODING = "base64"
OUT_MCAP_FILE_NAME = "json_log.mcap"
RLOG_FOLDER = "rlogs"
SCHEMAS_FOLDER = "schemas"
SCHEMA_EXTENSION = ".json"

schemas: dict[str, int] = {}
channels: dict[str, int] = {}
writer: Writer


def convertBytesToString(data):
  if isinstance(data, bytes):
    return data.decode('latin-1')  # Assuming UTF-8 encoding, adjust if needed
  elif isinstance(data, list):
    return [convertBytesToString(item) for item in data]
  elif isinstance(data, dict):
    return {key: convertBytesToString(value) for key, value in data.items()}
  else:
    return data


# Load jsonscheme for every Event
def loadSchema(schemaName):
  with open(os.path.join(SCHEMAS_FOLDER, schemaName + SCHEMA_EXTENSION), "r") as file:
    return json.loads(file.read())


# Foxglove creates one graph of an array, and not one for each item of an array
# This can be avoided by transforming array to separate objects
def transformListsToJsonDict(json_data):
  def convert_array_to_dict(array):
    new_dict = {}
    for index, item in enumerate(array):
      if isinstance(item, dict):
        new_dict[index] = transformListsToJsonDict(item)
      else:
        new_dict[index] = item
    return new_dict

  new_data = {}
  for key, value in json_data.items():
    if isinstance(value, list):
      new_data[key] = convert_array_to_dict(value)
    elif isinstance(value, dict):
      new_data[key] = transformListsToJsonDict(value)
    else:
      new_data[key] = value
  return new_data


# Transform openpilot thumbnail to foxglove compressedImage
def transformToFoxgloveSchema(jsonMsg):
  bytesImgData = jsonMsg.get("thumbnail").get("thumbnail").encode('latin1')
  base64ImgData = base64.b64encode(bytesImgData)
  base64_string = base64ImgData.decode('utf-8')
  foxMsg = {
    "timestamp": {"sec": "0", "nsec": jsonMsg.get("logMonoTime")},
    "frame_id": str(jsonMsg.get("thumbnail").get("frameId")),
    "data": base64_string,
    "format": "jpeg",
  }
  return foxMsg


# TODO: Check if there is a tool to build GEOJson
def transformMapCoordinates(jsonMsg):
  coordinates = []
  for jsonCoords in jsonMsg.get("navRoute").get("coordinates"):
    coordinates.append([jsonCoords.get("longitude"), jsonCoords.get("latitude")])

  # Define the GeoJSON
  geojson_data = {
    "type": "FeatureCollection",
    "features": [{"type": "Feature", "geometry": {"type": "LineString", "coordinates": coordinates}, "logMonoTime": jsonMsg.get("logMonoTime")}],
  }

  # Create the final JSON with the GeoJSON data encoded as a string
  geoJson = {"geojson": json.dumps(geojson_data)}

  return geoJson


def jsonToScheme(jsonData):
  zeroArray = False
  schema = {"type": "object", "properties": {}, "required": []}
  for key, value in jsonData.items():
    if isinstance(value, dict):
      tempScheme, zeroArray = jsonToScheme(value)
      if tempScheme == 0:
        return 0
      schema["properties"][key] = tempScheme
      schema["required"].append(key)
    elif isinstance(value, list):
      if all(isinstance(item, dict) for item in value) and len(value) > 0:  # Handle zero value arrays
        # Handle array of objects
        tempScheme, zeroArray = jsonToScheme(value[0])
        schema["properties"][key] = {"type": "array", "items": tempScheme if value else {}}
        schema["required"].append(key)
      else:
        if len(value) == 0:
          zeroArray = True
        # Handle array of primitive types
        schema["properties"][key] = {"type": "array", "items": {"type": "string"}}
        schema["required"].append(key)
    else:
      typeName = type(value).__name__
      if typeName == "str":
        typeName = "string"
      elif typeName == "bool":
        typeName = "boolean"
      elif typeName == "float":
        typeName = "number"
      elif typeName == "int":
        typeName = "integer"
      schema["properties"][key] = {"type": typeName}
      schema["required"].append(key)

  return schema, zeroArray


def saveScheme(scheme, schemaFileName):
  schemaFileName = schemaFileName + SCHEMA_EXTENSION
  # Create the new schemas folder
  os.makedirs(SCHEMAS_FOLDER, exist_ok=True)
  with open(os.path.join(SCHEMAS_FOLDER, schemaFileName), 'w') as json_file:
    json.dump(convertBytesToString(scheme), json_file)


def convertToFoxGloveFormat(jsonData, rlogTopic):
  jsonData["title"] = rlogTopic
  if rlogTopic == "thumbnail":
    jsonData = transformToFoxgloveSchema(jsonData)
    jsonData["title"] = FOXGLOVE_IMAGE_SCHEME_TITLE
  elif rlogTopic == "navRoute":
    jsonData = transformMapCoordinates(jsonData)
    jsonData["title"] = FOXGLOVE_GEOJSON_TITLE
  else:
    jsonData = transformListsToJsonDict(jsonData)
  return jsonData


def generateSchemas():
  listOfDirs = os.listdir(RLOG_FOLDER)
  # Open every dir in rlogs
  for directory in listOfDirs:
    # List every file in every rlog dir
    dirPath = os.path.join(RLOG_FOLDER, directory)
    listOfFiles = os.listdir(dirPath)
    lastIteration = len(listOfFiles)
    for iteration, file in enumerate(listOfFiles):
      # Load json data from every file until found one without empty arrays
      filePath = os.path.join(dirPath, file)
      with open(filePath, 'r') as jsonFile:
        jsonData = json.load(jsonFile)
        scheme, zerroArray = jsonToScheme(jsonData)
        # If array of len 0 has been found, type of its data can not be parsed, skip to the next log
        # in search for a non empty array. If there is not an non empty array in logs, put a dummy string type
        if zerroArray and not iteration == lastIteration - 1:
          continue
        title = jsonData.get("title")
        scheme["title"] = title
        # Add contentEncoding type, hardcoded in foxglove format
        if title == FOXGLOVE_IMAGE_SCHEME_TITLE:
          scheme["properties"]["data"]["contentEncoding"] = FOXGLOVE_IMAGE_ENCODING
        saveScheme(scheme, directory)
        break


def downloadLogs(logPaths):
  segment_counter = 0
  for logPath in logPaths:
    segment_counter += 1
    msg_counter = 1
    print(segment_counter)
    rlog = LogReader(logPath)
    for msg in rlog:
      jsonMsg = json.loads(json.dumps(convertBytesToString(msg.to_dict())))
      jsonMsg = convertToFoxGloveFormat(jsonMsg, msg.which())
      rlog_dir_path = os.path.join(RLOG_FOLDER, msg.which())
      if not os.path.exists(rlog_dir_path):
        os.makedirs(rlog_dir_path)
      file_path = os.path.join(rlog_dir_path, str(segment_counter) + "," + str(msg_counter))
      with open(file_path, 'w') as json_file:
        json.dump(jsonMsg, json_file)
      msg_counter += 1


def getLogMonoTime(jsonMsg):
  if jsonMsg.get("title") == FOXGLOVE_IMAGE_SCHEME_TITLE:
    logMonoTime = jsonMsg.get("timestamp").get("nsec")
  elif jsonMsg.get("title") == FOXGLOVE_GEOJSON_TITLE:
    logMonoTime = json.loads(jsonMsg.get("geojson")).get("features")[0].get("logMonoTime")
  else:
    logMonoTime = jsonMsg.get("logMonoTime")
  return logMonoTime


def processMsgs(args):
  msgFile, rlogTopicPath, rlogTopic = args
  msgFilePath = os.path.join(rlogTopicPath, msgFile)
  with open(msgFilePath, "r") as file:
    jsonMsg = json.load(file)
    logMonoTime = getLogMonoTime(jsonMsg)
    return {'channel_id': channels[rlogTopic], 'log_time': logMonoTime, 'data': json.dumps(jsonMsg).encode("utf-8"), 'publish_time': logMonoTime}


# Get logs from a path, and convert them into mcap
def createMcap(logPaths):
  print(f"Downloading logs [{len(logPaths)}]")
  downloadLogs(logPaths)
  print("Creating schemas")
  generateSchemas()
  print("Creating mcap file")

  listOfRlogTopics = os.listdir(RLOG_FOLDER)
  print(f"Registering schemas and channels [{len(listOfRlogTopics)}]")
  for counter, rlogTopic in enumerate(listOfRlogTopics):
    print(counter)
    schema = loadSchema(rlogTopic)
    schema_id = writer.register_schema(name=schema.get("title"), encoding="jsonschema", data=json.dumps(schema).encode())
    schemas[rlogTopic] = schema_id
    channel_id = writer.register_channel(schema_id=schemas[rlogTopic], topic=rlogTopic, message_encoding="json")
    channels[rlogTopic] = channel_id
    rlogTopicPath = os.path.join(RLOG_FOLDER, rlogTopic)
    msgFiles = os.listdir(rlogTopicPath)
    pool = Pool()
    results = pool.map(processMsgs, [(msgFile, rlogTopicPath, rlogTopic) for msgFile in msgFiles])
    pool.close()
    pool.join()
    for result in results:
      writer.add_message(channel_id=result['channel_id'], log_time=result['log_time'], data=result['data'], publish_time=result['publish_time'])


def is_program_installed(program_name):
  try:
    # Check if the program is installed using dpkg (for traditional Debian packages)
    subprocess.run(["dpkg", "-l", program_name], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return True
  except subprocess.CalledProcessError:
    # Check if the program is installed using snap
    try:
      subprocess.run(["snap", "list", program_name], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      return True
    except subprocess.CalledProcessError:
      return False


if __name__ == '__main__':
  # Example usage:
  program_name = "foxglove-studio"  # Change this to the program you want to check
  if is_program_installed(program_name):
    print(f"{program_name} detected.")
  else:
    print(f"{program_name} could not be detected.")
    installFoxglove = input("Would you like to install it? YES/NO? - ")
    if installFoxglove.lower() == "yes":
      try:
        subprocess.run(['./install_foxglove.sh'], check=True)
        print("Installation completed successfully.")
      except subprocess.CalledProcessError as e:
        print(f"Installation failed with return code {e.returncode}.")
  # Get a route
  if len(sys.argv) == 1:
    route_name = "a2a0ccea32023010|2023-07-27--13-01-19"
    print("No route was provided, using demo route")
  else:
    route_name = sys.argv[1]
  # Get logs for a route
  print("Getting route log paths")
  route = Route(route_name)
  logPaths = route.log_paths()
  # Start mcap writer
  with open(OUT_MCAP_FILE_NAME, "wb") as stream:
    writer = Writer(stream, compression=CompressionType.NONE)
    writer.start()
    createMcap(logPaths)
    writer.finish()
    print(f"File {OUT_MCAP_FILE_NAME} has been successfully created. Please import it into foxglove studio to continue.")
