<div align="center" style="text-align: center;">

<h1>MSGQ</h1>
<p><b>High-performance <a href="https://en.wikipedia.org/wiki/Publish%E2%80%93subscribe_pattern">pub/sub</a> messaging, made simple.<br>For Python, C, and C++.</b></p>

<h3>
  <a href="#quickstart">Quickstart</a>
  <span> · </span>
  <a href="#cheatsheet">Cheatsheet</a>
  <span> · </span>
  <a href="https://github.com/commaai/msgq/tree/master/msgq/examples">Examples</a>
  <span> · </span>
  <a href="https://discord.comma.ai">Discord</a>
</h3>

[![Discord](https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white)](https://discord.comma.ai)
[![PyPI](https://img.shields.io/pypi/v/msgq-ipc)](https://pypi.org/project/msgq-ipc/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/commaai/msgq)
[![Tests](https://github.com/commaai/msgq/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/commaai/msgq/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/commaai/msgq/blob/master/LICENSE)

</div>

---

MSGQ lets programs on the same machine exchange messages. A publisher sends messages to a named endpoint, and subscribers listen on that same endpoint. Each endpoint supports one publisher and multiple subscribers.

<p align="center">
  <img src="https://github.com/user-attachments/assets/79bb91cf-c9ad-4fb4-97d9-33359a083f0f" alt="1 KiB cross-process ping-pong benchmark"><br>
  <sub>1 KiB cross-process ping-pong on x86 Linux. <a href="https://github.com/commaai/msgq/blob/master/msgq/examples/benchmark.py">Benchmark script</a>.</sub>
</p>

## Quickstart

```sh
python -m pip install msgq-ipc
```

Run the included [publisher](https://github.com/commaai/msgq/blob/master/msgq/examples/publisher.py) and [subscriber](https://github.com/commaai/msgq/blob/master/msgq/examples/subscriber.py) examples in separate terminals:

```sh
python -m msgq.examples.publisher --endpoint demo   # terminal 1
python -m msgq.examples.subscriber --endpoint demo  # terminal 2
```

The subscriber prints `Hello from MSGQ!` once per second.

The core API sends and receives bytes:

```python
import msgq

publisher = msgq.pub_sock("hello")
subscriber = msgq.sub_sock("hello")
publisher.send(b"Hello from MSGQ!")
print(subscriber.receive())  # b'Hello from MSGQ!'
```

## Cheatsheet

```python
import msgq

# API reference; calls are not intended to run in sequence.

# === Create sockets ===
pub = msgq.pub_sock("demo")                       # One publisher per endpoint
sub = msgq.sub_sock("demo")                       # Receive messages from that endpoint
sub = msgq.sub_sock("demo", timeout=1000)         # Wait up to 1000 milliseconds per receive
sub = msgq.sub_sock("demo", conflate=True)        # Receive only the latest available message


# === Send & Receive ===
pub.send(b"hello")                               # Send nonempty bytes; returns None
sub.receive()                                    # Return bytes; block by default
sub.receive(non_blocking=True)                   # Return bytes immediately, or None
sub.setTimeout(1000)                             # Set receive timeout in milliseconds
sub.setTimeout(-1)                               # Restore indefinite blocking
msgq.drain_sock_raw(sub)                         # Return a list of all available messages
msgq.drain_sock_raw(sub, wait_for_one=True)      # Wait for the first message, then drain

# A receive timeout returns None; draining returns [] if no messages arrive.
# Slow subscribers can miss messages when the ring buffer wraps.

# === Poll multiple subscribers ===
poller = msgq.Poller()                           # Create a group of subscribers to watch
sub = msgq.sub_sock("demo", poller=poller)       # Create and register a subscriber
poller.registerSocket(sub)                       # Alternatively, register an existing socket
poller.poll(1000)                                # Return readable sockets; timeout in milliseconds
poller.poll(0)                                   # Check immediately; return [] if none are ready
poller.poll(-1)                                  # Wait indefinitely for a readable socket

# Register each socket once. Call receive() on the sockets returned by poll().

# === Reader synchronization and errors ===
pub.all_readers_updated()                        # Check whether tracked readers have caught up
pub.wait_for_readers(timeout=1.0, interval=0.01) # Wait for that condition; times are in seconds
msgq.IpcError                                    # Messaging failure exception
msgq.MultiplePublishersError                     # Publisher conflict; subclass of IpcError

# wait_for_readers() raises TimeoutError if its deadline expires.
# Synchronization checks queue positions, not application processing. It requires
# at least one tracked reader and ignores readers invalidated by an overwrite.
```

## Contributing

Issues and pull requests are welcome on [GitHub](https://github.com/commaai/msgq). Run `./test.sh` to build, lint, and test the package.

## License

MSGQ is available under the [MIT License](https://github.com/commaai/msgq/blob/master/LICENSE).

<details>
<summary>Under the hood</summary>

The message queue copies data on send and receive. A fake implementation is also available for deterministic testing.

### Storage
The storage for the queue consists of an area of metadata, and the actual buffer. The metadata contains:

1. A counter to the number of readers that are active
2. A pointer to the head of the queue for writing. From now on referred to as *write pointer*
3. A cycle counter for the writer. This counter is incremented when the writer wraps around
4. N pointers, pointing to the current read position for all the readers. From now on referred to as *read pointer*
5. N counters,  counting the number of cycles for all the readers
6. N booleans, indicating validity for all the readers. From now on referred to as *validity flag*

The counter and the pointer are both 32 bit values, packed into 64 bit so they can be read and written atomically.

The data buffer is a ring buffer. All messages are prefixed by an 8 byte size field, followed by the data. A size of -1 indicates a wrap-around, and means the next message is stored at the beginning of the buffer.


### Writing
Writing involves the following steps:

1. Check if the area that is to be written overlaps with any of the read pointers, mark those readers as invalid by clearing the validity flag.
2. Write the message
3. Increase the write pointer by the size of the message

In case there is not enough space at the end of the buffer, a special empty message with a prefix of -1 is written. The cycle counter is incremented by one. In this case step 1 will check there are no read pointers pointing to the remainder of the buffer. Then another write cycle will start with the actual message.

There always needs to be 8 bytes of empty space at the end of the buffer. By doing this there is always space to write the -1.

### Reset reader
When the reader is lagging too much behind the read pointer becomes invalid and no longer points to the beginning of a valid message. To reset a reader to the current write pointer, the following steps are performed:

1. Set valid flag
2. Set read cycle counter to that of the writer
3. Set read pointer to write pointer

### Reading
Reading involves the following steps:

1. Read the size field at the current read pointer
2. Read the validity flag
3. Copy the data out of the buffer
4. Increase the read pointer by the size of the message
5. Check the validity flag again

Before starting the copy, the valid flag is checked. This is to prevent a race condition where the size prefix was invalid, and the read could read outside of the buffer. Make sure that step 1 and 2 are not reordered by your compiler or CPU.

If a writer overwrites the data while it's being copied out, the data will be invalid. Therefore the validity flag is also checked after reading it. The order of step 4 and 5 does not matter.

If at steps 2 or 5 the validity flag is not set, the reader is reset. Any data that was already read is discarded. After the reader is reset, the reading starts from the beginning.

If a message with size -1 is encountered, step 3 and 4 are replaced by increasing the cycle counter and setting the read pointer to the beginning of the buffer. After that another read is performed.

</details>
