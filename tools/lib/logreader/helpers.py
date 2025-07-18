from urllib.parse import parse_qs, urlparse


def parse_indirect(identifier: str) -> str:
  if "useradmin.comma.ai" in identifier:
    query = parse_qs(urlparse(identifier).query)
    identifier = query["onebox"][0]
  elif "connect.comma.ai" in identifier:
    path = urlparse(identifier).path.strip("/").split("/")
    path = ['/'.join(path[:2]), *path[2:]]  # recombine log id

    identifier = path[0]
    if len(path) > 2:
      # convert url with seconds to segments
      start, end = int(path[1]) // 60, int(path[2]) // 60 + 1
      identifier = f"{identifier}/{start}:{end}"

      # add selector if it exists
      if len(path) > 3:
        identifier += f"/{path[3]}"
    else:
      # add selector if it exists
      identifier = "/".join(path)

  return identifier
