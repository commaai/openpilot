from cereal.messaging import SubMaster
from openpilot.common.params import Params

sm = SubMaster(["speechToText"])
import os
import re
import requests
import urllib.parse

def get_coordinates_from_transcript(transcript, proximity, mapbox_access_token):
  # Regular expression to find 'navigate to' or 'directions to' followed by an address
  pattern = r'\b(navigate to|directions to)\b\s+(.*?)(\.|$)'
  # Search for the pattern in the transcript
  match = re.search(pattern, transcript, re.IGNORECASE)
  if match:
    address = match.group(2).strip()
    encoded_address = urllib.parse.quote(address)
    mapbox_url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{encoded_address}.json?proximity={proximity}access_token={mapbox_access_token}"
    response = requests.get(mapbox_url)
    if response.status_code == 200:
      data = response.json()
      # Assuming the first result is the most relevant
      if data['features']:
        coordinates = {
          "latitude": data['features'][0]['geometry']['coordinates'][0],
          "longitude": data['features'][0]['geometry']['coordinates'][1],
        }
        return coordinates
      print("No coordinates")
    print(f"Mapbox API error: {response.status_code}")
  return False

def printer():
  params = Params()
  mapbox_access_token = os.environ["MAPBOX_TOKEN"]
  while True:
    dest = False
    transcript: str = ""
    sm.update(0)
    if sm.updated["speechToText"]:
      transcript = sm["speechToText"].result
      if not sm["speechToText"].finalResultReady:
        print(f'Interim result: {transcript}')
      else:
        print(f'Final result: {transcript}')
        proximity = params.get("LastGPSPosition")
        dest = get_coordinates_from_transcript(transcript,proximity, mapbox_access_token)
        if dest:
          params.put("NavDestination", dest)
          dest = False

if __name__ == "__main__":
    printer()

