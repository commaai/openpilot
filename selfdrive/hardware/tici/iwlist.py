import subprocess


def scan(interface="wlan0"):
  result = []
  try:
    r = subprocess.check_output(["iwlist", interface, "scan"], encoding='utf8')

    mac = None

    for line in r.split('\n'):
      if "Address" in line:
        # Add previous network in case no dBm signal level was seen
        if mac is not None:
          result.append({"mac": mac})
          mac = None

        mac = line.split(' ')[-1]
      elif "dBm" in line:
        try:
          i = line.index('Signal level=') + 13
          rss = int(line[i:].split(' ')[0])
          result.append({"mac": mac, "rss": rss})
          mac = None
        except ValueError:
          continue

    if mac is not None:
      result.append({"mac": mac})

    return result

  except Exception:
    return None


if __name__ == "__main__":
  import sys
  import pprint

  if len(sys.argv) > 1:
    pprint.pp(scan(sys.argv[1]))
  else:
    pprint.pp(scan())

