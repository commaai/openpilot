import argparse

import msgq


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Receive and print greetings from a publisher.")
  parser.add_argument("--endpoint", default="msgq_example", help="endpoint name (default: %(default)s)")
  args = parser.parse_args()
  subscriber = msgq.sub_sock(args.endpoint, timeout=1000)

  print("Ctrl-C to exit")
  try:
    while True:
      message = subscriber.receive()
      if message is not None:
        print(f"Received: {message.decode('utf-8')}", flush=True)
  except KeyboardInterrupt:
    pass
