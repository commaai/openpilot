import sys
import json
import base64
from openpilot.tools.lib.route import Route
from openpilot.tools.lib.logreader import LogReader
from mcap.writer import Writer

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
    with open("./schemas/" + schemaName + ".json", "r") as file:
        return file.read()

# Foxglove creates one graph of an array, and not one for each item of an array
# This can be avoided by transforming array to separate values
def transform_json(json_data, arr_key):
    newTempC = {}
    counter = 0
    for tempC in json_data.get(arr_key):
        newTempC[counter] = tempC
        counter+=1
    json_data[arr_key] = newTempC
    return json_data

def transformToFoxgloveSchema(jsonMsg):
    bytesImgData = jsonMsg.get("thumbnail").get("thumbnail").encode('latin1')
    base64ImgData = base64.b64encode(bytesImgData)
    base64_string = base64ImgData.decode('utf-8')
    foxMsg = {
        "timestamp":{
            "sec":"0",
            "nsec":jsonMsg.get("logMonoTime")
        },
        "frame_id":str(jsonMsg.get("thumbnail").get("frameId")),
        "data": base64_string,
        "format": "jpeg"
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
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates
                }
            }
        ]
    }

    # Create the final JSON with the GeoJSON data encoded as a string
    geoJson = {
        "geojson": json.dumps(geojson_data)
    }

    return geoJson

# Get logs from a path, and convert them into mcap
def createMcap(logPaths):
    segment_counter = 0
    for logPath in logPaths:
        segment_counter+=1
        print(segment_counter)
        # if segment_counter != 2:
        #     continue
        rlog = LogReader(logPath)
        for msg in rlog:
            jsonMsg = json.loads(json.dumps(convertBytesToString(msg.to_dict())))
            if msg.which() == "thumbnail":
                jsonMsg = transformToFoxgloveSchema(jsonMsg)
            elif msg.which() == "deviceState":
                jsonMsg["deviceState"] = transform_json(jsonMsg.get("deviceState"), "cpuTempC")
            elif msg.which() == "navRoute":
                jsonMsg = transformMapCoordinates(jsonMsg)
            if msg.which() not in schemas:
                schema = loadSchema(msg.which())
                schema_id = writer.register_schema(
                    name= json.loads(schema).get("title"),
                    encoding="jsonschema",
                    data=schema.encode()
                )
                schemas[msg.which()] = schema_id
            if msg.which() not in channels:
                channel_id = writer.register_channel(
                    schema_id= schemas[msg.which()],
                    topic=msg.which(),
                    message_encoding="json"
                )
                channels[msg.which()] = channel_id
            writer.add_message(
                channel_id=channels[msg.which()],
                log_time=msg.logMonoTime,
                data=json.dumps(jsonMsg).encode("utf-8"),
                publish_time=msg.logMonoTime
            )
            # if msg.which() == "navRoute":
            #     print(json.dumps(jsonMsg))
            #     exit(0)

# TODO: Check if foxglove is installed
if __name__ == '__main__':
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
    with open("json_log.mcap", "wb") as stream:
        writer = Writer(stream)
        writer.start()
        createMcap(logPaths)
        writer.finish()
