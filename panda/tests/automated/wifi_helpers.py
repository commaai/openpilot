import os
FNULL = open(os.devnull, 'w')
def _connect_wifi(dongle_id, pw, insecure_okay=False):
  ssid = "panda-" + dongle_id

  r = subprocess.call(["ping", "-W", "4", "-c", "1", "192.168.0.10"], stdout=FNULL, stderr=subprocess.STDOUT)
  if not r:
    # Can already ping, try connecting on wifi
    try:
      p = Panda("WIFI")
      p.get_serial()
      print("Already connected")
      return
    except:
      pass

  print("WIFI: connecting to %s" % ssid)

  while 1:
    if sys.platform == "darwin":
      os.system("networksetup -setairportnetwork en0 %s %s" % (ssid, pw))
    else:
      wlan_interface = subprocess.check_output(["sh", "-c", "iw dev | awk '/Interface/ {print $2}'"]).strip().decode('utf8')
      cnt = 0
      MAX_TRIES = 10
      while cnt < MAX_TRIES:
        print("WIFI: scanning %d" % cnt)
        os.system("iwlist %s scanning > /dev/null" % wlan_interface)
        os.system("nmcli device wifi rescan")
        wifi_networks = [x.decode("utf8") for x in subprocess.check_output(["nmcli","dev", "wifi", "list"]).split(b"\n")]
        wifi_scan = [x for x in wifi_networks if ssid in x]
        if len(wifi_scan) != 0:
          break
        time.sleep(0.1)
        # MAX_TRIES tries, ~10 seconds max
        cnt += 1
      assert cnt < MAX_TRIES
      if "-pair" in wifi_scan[0]:
        os.system("nmcli d wifi connect %s-pair" % (ssid))
        connect_cnt = 0
        MAX_TRIES = 100
        while connect_cnt < MAX_TRIES:
          connect_cnt += 1
          r = subprocess.call(["ping", "-W", "4", "-c", "1", "192.168.0.10"], stdout=FNULL, stderr=subprocess.STDOUT)
          if r:
            print("Waiting for panda to ping...")
            time.sleep(0.5)
          else:
            break
        if insecure_okay:
          break
        # fetch webpage
        print("connecting to insecure network to secure")
        try:
          r = requests.get("http://192.168.0.10/")
        except requests.ConnectionError:
          r = requests.get("http://192.168.0.10/")
        assert r.status_code==200

        print("securing")
        try:
          r = requests.get("http://192.168.0.10/secure", timeout=0.01)
        except requests.exceptions.Timeout:
          print("timeout http request to secure")
          pass
      else:
        ret = os.system("nmcli d wifi connect %s password %s" % (ssid, pw))
        if os.WEXITSTATUS(ret) == 0:
          #check ping too
          ping_ok = False
          connect_cnt = 0
          MAX_TRIES = 10
          while connect_cnt < MAX_TRIES:
            connect_cnt += 1
            r = subprocess.call(["ping", "-W", "4", "-c", "1", "192.168.0.10"], stdout=FNULL, stderr=subprocess.STDOUT)
            if r:
              print("Waiting for panda to ping...")
              time.sleep(0.1)
            else:
              ping_ok = True
              break
          if ping_ok:
            break

  # TODO: confirm that it's connected to the right panda