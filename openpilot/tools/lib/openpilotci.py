BASE_URL = "https://commadataci.blob.core.windows.net/openpilotci/"

def get_url(route_name: str, segment_num, filename: str) -> str:
  return BASE_URL + f"{route_name.replace('|', '/')}/{segment_num}/{filename}"
