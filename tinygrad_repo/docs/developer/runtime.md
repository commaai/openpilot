# Runtime Overview

## Overview

A typical runtime consists of the following parts:

- [Compiled](#compiled)
- [Allocator](#allocator)
- [Program](#program)
- [Compiler](#compiler)

### Compiled

The `Compiled` class is responsible for initializing and managing a device.

::: tinygrad.device.Compiled
    options:
        members: [
            "synchronize"
        ]
        show_source: false

### Allocator

The `Allocator` class is responsible for managing memory on the device. There is also a version called the `LRUAllocator`, which caches allocated buffers to optimize performance.

::: tinygrad.device.Allocator
    options:
        members: true
        show_source: false

::: tinygrad.device.LRUAllocator
    options:
        members: true
        show_source: false

### Program

The `Program` class is created for each loaded program. It is responsible for executing the program on the device. As an example, here is a `CPUProgram` implementation which loads program and runs it.

::: tinygrad.runtime.ops_cpu.CPUProgram
    options:
        members: true

### Compiler

The `Compiler` class compiles the output from the `Renderer` and produces it in a device-specific format.

::: tinygrad.device.Compiler
    options:
        members: true
