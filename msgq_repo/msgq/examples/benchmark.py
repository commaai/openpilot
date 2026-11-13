# /// script
# requires-python = ">=3.11"
# dependencies = ["msgq-ipc>=1.0", "pyzmq", "eclipse-zenoh>=1.10", "lcm", "matplotlib"]
# [tool.uv.sources]
# msgq-ipc = { path = "../.." }
# [tool.ty.rules]
# unresolved-import = "ignore"
# ///

import os
import lcm
import zmq
import msgq
import time
import zenoh
import random
import argparse
import platform
import tempfile
import matplotlib
import statistics
import multiprocessing

from pathlib import Path
from threading import Event
from collections import deque  # codespell:ignore deque
from functools import partial
from contextlib import ExitStack

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BACKENDS = ("msgq", "pyzmq IPC", "Pipe bytes", "zenoh IPC", "LCM UDP")
ROUNDS = 1000
SECONDS = 2
REPEATS = 5
TIMEOUT = 60


def setup_backend(name, role, directory, prefix, barrier, pipe, cleanup, size):
  outgoing, incoming = ("request", "reply") if role == "sender" else ("reply", "request")
  if name == "msgq":
    os.environ["OPENPILOT_PREFIX"] = prefix
    publisher = msgq.pub_sock(outgoing)
    # Creating a publisher resets its queue, so both must exist before subscribing.
    barrier.wait(10)
    subscriber = msgq.sub_sock(incoming)
    return publisher.send, subscriber.receive
  if name == "pyzmq IPC":
    context = cleanup.enter_context(zmq.Context())
    publisher = context.socket(zmq.XPUB)
    subscriber = context.socket(zmq.SUB)
    cleanup.callback(publisher.close, linger=0)
    cleanup.callback(subscriber.close, linger=0)
    publisher.setsockopt(zmq.RCVTIMEO, 5000)
    subscriber.setsockopt(zmq.RCVTIMEO, 5000)
    subscriber.setsockopt(zmq.SUBSCRIBE, b"")
    publisher.bind(f"ipc://{directory}/{outgoing}")
    subscriber.connect(f"ipc://{directory}/{incoming}")
    if publisher.recv() != b"\x01":
      raise RuntimeError("Unexpected ZeroMQ subscription")
    return publisher.send, subscriber.recv
  if name == "Pipe bytes":
    cleanup.enter_context(pipe)
    return pipe.send_bytes, pipe.recv_bytes
  if name == "zenoh IPC":
    endpoint = f"unixsock-stream/{directory}/zenoh"
    config = zenoh.Config()
    config.insert_json5("listen/endpoints", repr([endpoint] if role == "sender" else []))
    config.insert_json5("connect/endpoints", repr([] if role == "sender" else [endpoint]))
    config.insert_json5("scouting/multicast/enabled", "false")
    config.insert_json5("scouting/gossip/enabled", "false")
    config.insert_json5("transport/unicast/qos/enabled", "false")
    # Low-latency transport cannot fragment a 64 KiB payload.
    config.insert_json5("transport/unicast/lowlatency", str(size <= 1024).lower())
    session = cleanup.enter_context(zenoh.open(config))
    zenoh_subscriber = session.declare_subscriber(incoming)
    cleanup.callback(zenoh_subscriber.undeclare)
    publisher = session.declare_publisher(outgoing)
    cleanup.callback(publisher.undeclare)
    ready = Event()
    listener = publisher.declare_matching_listener(lambda status: ready.set() if status.matching else None)
    cleanup.callback(listener.undeclare)
    if publisher.matching_status.matching:
      ready.set()
    if not ready.wait(10):
      raise RuntimeError("Zenoh subscriber did not become ready")
    return publisher.put, lambda: zenoh_subscriber.recv().payload.to_bytes()
  if name == "LCM UDP":
    bus = lcm.LCM("udpm://239.255.76.67:7667?ttl=0")
    messages = deque()  # codespell:ignore deque
    token = Path(directory).name
    subscriber = bus.subscribe(token + incoming, lambda channel, data: messages.append(data))
    cleanup.callback(bus.unsubscribe, subscriber)

    def receive():
      while not messages:
        if bus.handle_timeout(5000) == 0:
          raise TimeoutError("LCM receive timed out; check host multicast support")
      return messages.popleft()

    return partial(bus.publish, token + outgoing), receive
  raise ValueError(name)


def payload(sequence, padding):
  return sequence.to_bytes(8, "little") + padding


def sender(send, receive, size):
  padding = b"x" * (size - 8)
  sequence = 0

  def exchange():
    nonlocal sequence
    message = payload(sequence, padding)
    send(message)
    reply = receive()
    if reply != message:
      raise RuntimeError(f"Invalid reply at sequence {sequence}: {None if reply is None else (len(reply), reply[:8])}")
    sequence += 1

  start = time.perf_counter()
  while sequence < 100 or time.perf_counter() - start < 0.1:
    exchange()
  send(b"BEGIN")
  if receive() != b"BEGIN":
    raise RuntimeError("Invalid measurement handshake")
  warm_up_count = sequence
  start = time.perf_counter()
  while True:
    for _ in range(64):
      exchange()
    elapsed = time.perf_counter() - start
    count = sequence - warm_up_count
    if count >= ROUNDS and elapsed >= SECONDS:
      break
  # Stop timing before the completion handshake.
  send(b"END")
  if receive() != b"END":
    raise RuntimeError("Invalid completion handshake")
  return {"round_trips": count, "seconds": elapsed}


def echo(send, receive, size):
  padding = b"x" * (size - 8)
  sequence = 0
  measured_start = None
  while True:
    message = receive()
    if message == b"BEGIN" and measured_start is None:
      measured_start = sequence
      send(message)
    elif message == b"END" and measured_start is not None:
      send(message)
      return {"round_trips": sequence - measured_start}
    else:
      if message != payload(sequence, padding):
        raise RuntimeError(f"Invalid request at sequence {sequence}: {None if message is None else (len(message), message[:8])}")
      sequence += 1
      send(message)


def peer(name, role, directory, prefix, barrier, pipe, control, size):
  try:
    with ExitStack() as cleanup:
      send, receive = setup_backend(name, role, directory, prefix, barrier, pipe, cleanup, size)
      barrier.wait(10)
      result = sender(send, receive, size) if role == "sender" else echo(send, receive, size)
    control.send(("ok", result))
  except BaseException as error:
    control.send(("error", f"{name} {role}: {type(error).__name__}: {error}"))
  finally:
    pipe.close()
    control.close()


def measure(name, size):
  context = multiprocessing.get_context("spawn")
  shared_root = "/tmp" if platform.system() == "Darwin" else "/dev/shm"
  with tempfile.TemporaryDirectory(prefix="mq_", dir="/tmp") as directory, \
       tempfile.TemporaryDirectory(prefix="msgq_", dir=shared_root) as shared:
    # An owned prefix isolates and removes MSGQ files without touching existing queues.
    prefix = Path(shared).name.removeprefix("msgq_")
    barrier = context.Barrier(2)
    pipes = context.Pipe()
    processes, controls = [], []
    deadline = time.monotonic() + TIMEOUT
    try:
      for role, pipe in zip(("sender", "echo"), pipes, strict=True):
        parent, child = context.Pipe(duplex=False)
        controls.append(parent)
        process = context.Process(target=peer, args=(name, role, directory, prefix, barrier, pipe, child, size))
        process.start()
        processes.append(process)
        child.close()
      for pipe in pipes:
        pipe.close()
      results = {}
      while len(results) < 2:
        if time.monotonic() >= deadline:
          raise TimeoutError(f"{name}: trial exceeded {TIMEOUT}s")
        for index, control in enumerate(controls):
          if index in results:
            continue
          if control.poll(0.05):
            try:
              status, result = control.recv()
            except EOFError as error:
              raise RuntimeError(f"{name}: peer {index} closed its result pipe unexpectedly") from error
            if status != "ok":
              raise RuntimeError(result)
            results[index] = result
          elif processes[index].exitcode is not None:
            raise RuntimeError(f"{name}: peer exited without results ({processes[index].exitcode})")
      for process in processes:
        process.join(max(0, deadline - time.monotonic()))
        if process.exitcode != 0:
          raise RuntimeError(f"{name}: peer failed to exit cleanly")
      result = results[0]
      if result["round_trips"] != results[1]["round_trips"]:
        raise RuntimeError(f"{name}: peers disagree on verified message count")
      return result
    finally:
      for process in processes:
        if process.is_alive():
          process.terminate()
      for process in processes:
        process.join(2)
        if process.is_alive():
          process.kill()
          process.join()
      for connection in (*controls, *pipes):
        connection.close()


def plot_results(results, path):
  from matplotlib.ticker import EngFormatter, MaxNLocator

  labels = {"msgq": "msgq", "pyzmq IPC": "pyzmq", "Pipe bytes": "multiprocessing.Pipe", "zenoh IPC": "zenoh", "LCM UDP": "LCM"}
  rows = sorted(((name, rate) for name, size, rate in results if size == 1024), key=lambda row: row[1])
  maximum = max(rate for _, rate in rows)
  formatter = EngFormatter(sep="", places=2)
  with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"]}):
    fig, ax = plt.subplots(figsize=(10, max(2.2, len(rows) * 0.48 + 0.7)))
    fig.set_facecolor("white")
    ax.set(xlim=(0, maximum * 1.2), ylim=(len(rows) - 0.5, -0.5), yticks=[])
    # Keep grid lines out of the values on the right.
    ticks = MaxNLocator(nbins=4).tick_values(0, maximum)
    ax.set_xticks([tick for tick in ticks if 0 <= tick < maximum])
    ax.xaxis.set_major_formatter(EngFormatter(sep=""))
    ax.set_xlabel("messages/sec — higher is better", fontsize=14, labelpad=8, color="#172b3a")
    ax.tick_params(length=0, pad=6, labelsize=11, colors="#425b6c")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#b7b7b7", linewidth=1.5)
    for side, spine in ax.spines.items():
      spine.set(visible=side == "left", color="#192a32", linewidth=2)
    for index, (name, rate) in enumerate(rows):
      gray = str(0.88 - 0.28 * index / max(len(rows) - 2, 1))
      ax.barh(index, rate, height=1, color="#31cf43" if name == "msgq" else gray, zorder=2)
      ax.text(maximum * 0.029, index, labels[name], va="center", fontsize=16, fontweight="bold", color="#172b3a")
      ax.text(maximum * 1.173, index, formatter(rate), ha="right", va="center", fontsize=16, fontweight="bold", color="#172b3a")
    fig.subplots_adjust(left=0.055, right=0.98, top=0.98, bottom=0.21)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
  print(f"Chart saved to {path}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Cross-process pub/sub ping-pong benchmark.")
  parser.add_argument("--plot", type=Path, help="save a 1 KiB chart")
  args = parser.parse_args()
  if "CEREAL_FAKE" in os.environ:
    parser.error("Unset CEREAL_FAKE to benchmark the real MSGQ backend")
  print("Cross-process ping-pong; messages/sec counts requests and replies.", flush=True)
  samples = {(name, size): [] for name in BACKENDS for size in (64, 1024, 65536)}
  randomizer = random.Random(0)
  for _ in range(REPEATS):
    jobs = list(samples)
    randomizer.shuffle(jobs)
    for name, size in jobs:
      sample = measure(name, size)
      rate = 2 * sample["round_trips"] / sample["seconds"]
      samples[name, size].append(rate)
      print(f"{name:<12} {size:>6} bytes: {rate:>12,.0f} messages/sec", flush=True)
  results = [(name, size, statistics.median(rates)) for (name, size), rates in samples.items()]
  print("\nMedians:")
  for name, size, rate in results:
    print(f"{name:<12} {size:>6} bytes: {rate:>12,.0f} messages/sec")
  if args.plot:
    plot_results(results, args.plot)
