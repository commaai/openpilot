## Intro

Remu is an RDNA3 emulator built to test correctness of RDNA3 code. It is used in [tinygrad's AMD CI](https://github.com/tinygrad/tinygrad).

Most of the common instructions are implemented, but some formats like IMG are not supported.

Remu is only for testing correctness of program output, it is not a cycle accurate simulator.

## Build Locally

Remu is written in Rust. Make sure you have [Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html).

To build the project, run:

```bash
cargo build --release --manifest-path ./extra/remu/Cargo.toml
```

This will produce a binary in the `extra/remu/target/release` directory.

## Usage with tinygrad

The latest binaries are released in https://github.com/Qazalin/remu/releases. Alternatively, you can [build locally](#build-locally).

Tinygrad does not yet output RDNA3 kernels directly. You can either install comgr or use `AMD_LLVM=1` (default) if you have [LLVM@19](https://github.com/tinygrad/tinygrad/blob/e2ed673c946c8f1774d816c75e52a994c2dd8a88/.github/actions/setup-tinygrad/action.yml#L208).

`PYTHONPATH="." MOCKGPU=1 AMD=1 python test/test_tiny.py TestTiny.test_plus` runs an emulated RDNA3 kernel with Remu.

Add `DEBUG=6` to see Remu's logs.

### DEBUG output

Remu runs each thread one at a time in a nested for loop, see lib.rs. The DEBUG output prints information about the current thread.

The DEBUG output has 3 sections:

```
<------------ 1 ----------> <--- 2 ---> <--------------------------------------- 3 ------------------------------------------>
[0   0   0  ] [0   0   0  ] 0  F4080100 SMEM { op: 2, sdata: 4, sbase: 0, offset: 0, soffset: 124, glc: false, dlc: false }
```

#### Section 1: Grid info

`[gid.x, gid.y, gid.z], [lid.x, lid.y, lid.z]` of the current thread.

#### Section 2: Wave info

`<lane> <instruction hex>`

RDNA3 divides threads into chunks of 32. Each thread is assigned to a "lane" from 0-31.

In Remu, even though all threads run one at a time, each 32 thread chunk (a wave) shares state like SGPR, VGPR, LDS, EXEC mask, etc.
Remu can simulate up to one wave sync instruction.
For more details, see work_group.rs.

Section 2 can have a green or gray color.

Green = The thread is actively executing the instruction.

Gray = The thread has been "turned off" by the EXEC mask, it skips execution of some instructions. (refer to "EXECute Mask" on [page 23](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf#page=23) of ISA docs for more details.)

To see the colors in action, try running `DEBUG=6 PYTHONPATH="." MOCKGPU=1 AMD=1 python test/test_ops.py TestOps.test_arange_big`. See how only lane 0 writes to global memory:
```
[255 0   0  ] [0   0   0  ] 0  DC6A0000 FLAT { op: 26, offset: 0, dlc: false, glc: false, slc: false, seg: 2, addr: 8, data: 0, saddr: 0, sve: false, vdst: 0 }
[255 0   0  ] [1   0   0  ] 1  DC6A0000
[255 0   0  ] [2   0   0  ] 2  DC6A0000
[255 0   0  ] [3   0   0  ] 3  DC6A0000
[255 0   0  ] [3   0   0  ] 4  DC6A0000
```

#### Section 3: Decoded Instruction

This prints the instruction type and all the parsed bitfields.

Remu output vs llvm-objdump:

```
s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000160C: F4040000 F8000010
SMEM { op: 1, sdata: 0, sbase: 0, offset: 16, soffset: 124, glc: false, dlc: false }
```
