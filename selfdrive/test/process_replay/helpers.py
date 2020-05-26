import bz2

def save_log(dest, log_msgs):
  dat = b""
  for msg in log_msgs:
    dat += msg.as_builder().to_bytes()
  dat = bz2.compress(dat)

  with open(dest, "wb") as f:
   f.write(dat)

