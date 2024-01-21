from cereal import messaging, log
from openpilot.common.params import Params
import json

STTState = log.SpeechToText.State

sm = messaging.SubMaster(["speechToText"])
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
    mapbox_url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{encoded_address}.json?access_token=pk.eyJ1IjoicnlsZXltY2MiLCJhIjoiY2xjeDl5aGp4MTBmeDNzb2Vua2QyNWN1bSJ9.CrbD-j1LQkBdOqyWcZneyQ"
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

def main():
  params = Params()
  mapbox_access_token = os.environ["MAPBOX_TOKEN"]
  while True:
    dest = False
    transcript: str = ""
    sm.update(0)
    if sm.updated["speechToText"]:
      transcript = sm["speechToText"].transcript
      if not sm["speechToText"].state == log.SpeechToText.State.final:
        print(f'Interim result: {transcript}')
      else:
        print(f'Final result: {transcript}')
        proximity = params.get("LastGPSPosition")
        print(proximity)
        dest = get_coordinates_from_transcript(transcript,proximity, mapbox_access_token)
        if dest:
          params.put("NavDestination", json.dumps(dest))
          print(dest)
          dest = False

if __name__ == "__main__":
    main()

