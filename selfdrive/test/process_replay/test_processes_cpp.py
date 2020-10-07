from tools.lib.logreader import LogReader

if __name__ == "__main__":
  path = "https://commadataci.blob.core.windows.net/openpilotci/0375fdf7b1ce594d|2019-06-13--08-32-25--3_ubloxd_28b671c1cae9a4fca38c0230591b735edaa58ef9.bz2"
  lr = LogReader(path)
  #print(len(list(lr)))
  for x in lr:
    print(x)
