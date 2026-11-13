import argparse
import time

import msgq


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Publish a greeting once per second.")
  parser.add_argument("--endpoint", default="msgq_example", help="endpoint name (default: %(default)s)")
  args = parser.parse_args()
  publisher = msgq.pub_sock(args.endpoint)

  print("Ctrl-C to exit")
  try:
    while True:
      message = "Hello from MSGQ!"
      publisher.send(message.encode("utf-8"))
      print(f"Sent: {message}", flush=True)
      time.sleep(1)
  except KeyboardInterrupt:
    pass
