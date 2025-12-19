# HCQ Compatible Runtime

## Overview

The main aspect of HCQ-compatible runtimes is how they interact with devices. In HCQ, all interactions with devices occur in a hardware-friendly manner using [command queues](#command-queues). This approach allows commands to be issued directly to devices, bypassing runtime overhead such as HIP or CUDA. Additionally, by using the HCQ API, these runtimes can benefit from various optimizations and features, including [HCQGraph](#hcqgraph) and built-in profiling capabilities.

### Command Queues

To interact with devices you create a `HWQueue`. Some methods are required, like timestamp and synchronization methods like [signal](#tinygrad.runtime.support.hcq.HWQueue.signal) and [wait](#tinygrad.runtime.support.hcq.HWQueue.wait), while others are dependent on it being a compute or copy queue.

For example, the following Python code enqueues a wait, execute, and signal command on the HCQ-compatible device:
```python
HWQueue().wait(signal_to_wait, value_to_wait) \
         .exec(program, args_state, global_dims, local_dims) \
         .signal(signal_to_fire, value_to_fire) \
         .submit(your_device)
```

Each runtime should implement the required functions that are defined in the `HWQueue` classes.

::: tinygrad.runtime.support.hcq.HWQueue
    options:
        members: [
            "signal",
            "wait",
            "timestamp",
            "bind",
            "submit",
            "memory_barrier",
            "exec",
            "copy",
        ]
        show_source: false

### HCQ Compatible Device

The `HCQCompiled` class defines the API for HCQ-compatible devices. This class serves as an abstract base class that device-specific implementations should inherit from and implement.

::: tinygrad.runtime.support.hcq.HCQCompiled
    options:
        show_source: false

#### Signals

Signals are device-dependent structures used for synchronization and timing in HCQ-compatible devices. They should be designed to record both a `value` and a `timestamp` within the same signal. HCQ-compatible backend implementations should use `HCQSignal` as a base class.

::: tinygrad.runtime.support.hcq.HCQSignal
    options:
        members: [value, timestamp, wait]
        show_source: false

The following Python code demonstrates the usage of signals:

```python
signal = your_device.new_signal(value=0)

HWQueue().timestamp(signal) \
         .signal(signal, value_to_fire) \
         .submit(your_device)

signal.wait(value_to_fire)
signaled_value = signal.value # should be the same as `value_to_fire`
timestamp = signal.timestamp
```

##### Synchronization signals

Each HCQ-compatible device must allocate two signals for global synchronization purposes. These signals are passed to the `HCQCompiled` base class during initialization: an active timeline signal `self.timeline_signal` and a shadow timeline signal `self._shadow_timeline_signal` which helps to handle signal value overflow issues. You can find more about synchronization in the [synchronization section](#synchronization)

### HCQ Compatible Allocator

The `HCQAllocator` base class simplifies allocator logic by leveraging [command queues](#command-queues) abstractions. This class efficiently handles copy and transfer operations, leaving only the alloc and free functions to be implemented by individual backends.

::: tinygrad.runtime.support.hcq.HCQAllocator
    options:
        members: [
            "_alloc",
            "_free",
        ]
        show_source: false

#### HCQ Allocator Result Protocol

Backends must adhere to the `HCQBuffer` protocol when returning allocation results.

::: tinygrad.runtime.support.hcq.HCQBuffer
    options:
        members: true
        show_source: false

### HCQ Compatible Program

`HCQProgram` is a base class for defining programs compatible with HCQ-enabled devices. It provides a flexible framework for handling different argument layouts (see `HCQArgsState`).

::: tinygrad.runtime.support.hcq.HCQProgram
    options:
        members: true
        show_source: false

#### Arguments State

`HCQArgsState` is a base class for managing the argument state for HCQ programs. Backend implementations should create a subclass of `HCQArgsState` to manage arguments for the given program.

::: tinygrad.runtime.support.hcq.HCQArgsState
    options:
        members: true
        show_source: false

**Lifetime**: The `HCQArgsState` is passed to `HWQueue.exec` and is guaranteed not to be freed until `HWQueue.submit` for the same queue is called.

### Synchronization

HCQ-compatible devices use a global timeline signal for synchronizing all operations. This mechanism ensures proper ordering and completion of tasks across the device. By convention, `self.timeline_value` points to the next value to signal. So, to wait for all previous operations on the device to complete, wait for `self.timeline_value - 1` value. The following Python code demonstrates the typical usage of signals to synchronize execution to other operations on the device:

```python
HWQueue().wait(your_device.timeline_signal, your_device.timeline_value - 1) \
         .exec(...)
         .signal(your_device.timeline_signal, your_device.next_timeline()) \
         .submit(your_device)

# Optionally wait for execution
your_device.timeline_signal.wait(your_device.timeline_value - 1)
```

## HCQGraph

[HCQGraph](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/runtime/graph/hcq.py) is a core feature that implements `GraphRunner` for HCQ-compatible devices. `HCQGraph` builds static `HWQueue` for all operations per device. To optimize enqueue time, only the necessary parts of the queues are updated for each run using the symbolic variables, avoiding a complete rebuild.
Optionally, queues can implement a `bind` API, which allows further optimization by eliminating the need to copy the queues into the device ring.
