Google's TPU
--------------------------------------------------------------------

We document the Google TPU v2/v3 in order to support it in tinygrad without the XLA compiler.

## Creating a Google Cloud TPU VM

This costs $4.50/hr for a TPUv2-8 machine, the cheapest VM.

```bash
gcloud alpha compute tpus tpu-vm create test --zone=us-central1-b --accelerator-type=v2-8 --version=v2-alpha
gcloud alpha compute tpus tpu-vm ssh test --zone us-central1-b
# and for when you are done
gcloud alpha compute tpus tpu-vm delete test --zone us-central1-b
gcloud alpha compute tpus tpu-vm list --zone us-central1-b
```

Aside from the usual VM stuff, there's 4 accelerators on the PCI-E bus. (v2-8 is 4 chips with 2 cores each)

```
# lspci
00:04.0 Unassigned class [ff00]: Google, Inc. Device 0027
00:05.0 Unassigned class [ff00]: Google, Inc. Device 0027
00:06.0 Unassigned class [ff00]: Google, Inc. Device 0027
00:07.0 Unassigned class [ff00]: Google, Inc. Device 0027
```

They show up in `/sys/class/accel` (tons of files here) and the driver lives in `/lib/libtpu.so`. The devices are in `/dev/accel[0-3]`, and a bunch of stuff is mmaped. They are "ba16c7433" chips.

We grab the minimal TPU [example from TensorFlow](https://github.com/tensorflow/tensorflow/blob/695b4c93d5da7277eb845937b79b66f9f363ed94/tensorflow/compiler/xla/python/tpu_driver/client/libtpu_client.c). When the compiler runs, it produces tons of great logs in `/tmp/tpu_logs`

```bash
cd tfexample
gcc -o libtpu_client libtpu_client.c -ltpu
TPU_VLOG_LEVEL=99 ./libtpu_client
```

From these logs, we find the "LLO Instructions"

## VLIW Instruction (322b VLIW bundle)

```
  spare         : 0   (0,1)
  vex_mxu       : 0   (1,1)
* 1 misc slot
  msc_targ      : 0   (2,3)
  msc_opnd      : 0   (5,3)
  msc_op        : 0   (8,5)
  msc_pred      : 31  (13,5)
* 2 matrix slots (push, pop)
  vres_dest     : 28  (18,2)
  vres_op       : 28  (20,2)
  vres_pred     : 31  (22,5)
  vex_source    : 28  (27,2)
  vex_subop     : 24  (29,3)
  vex_op        : 24  (32,3)
  vex_pred      : 31  (35,5)
* 4 vector slots (2 for load/store)
  vld_ttu       : 30  (40,1)
  vld_stride    : 24  (41,3)
  vld_offset    : 24  (44,2)
  vld_base      : 24  (46,2)
  vld_submsk    : 24  (48,3)
  vld_dest      : 0   (51,5)
  vld_op        : 0   (56,2)
  vld_pred      : 31  (58,5)
  vst_ttu       : 30  (63,1)
  vst_iar       : 30  (64,1)
  vst_value_two : 24  (65,3)
  vst_offset    : 24  (68,2)
  vst_base      : 24  (70,2)
  vst_value_one : 24  (72,3)
  vst_source    : 0   (75,5)
  vst_op        : 0   (80,5)
  vst_pred      : 31  (85,5)
* 4 vector slots (2 for ALU)
  v1_dest       : 0   (90,5)
  v1_y_vreg     : 0   (95,5)
  v1_y_src      : 0   (100,5)
  v1_x          : 0   (105,5)
  v1_op         : 0   (110,6)
  v1_pred       : 31  (116,5)
  v0_dest       : 0   (121,5)
  v0_y_vreg     : 0   (126,5)
  v0_y_src      : 0   (131,5)
  v0_x          : 0   (136,5)
  v0_op         : 0   (141,6)
  v0_pred       : 31  (147,5)
* 3 scalar registers copied in to the vector units?
  vs2           : 0   (152,5)
  vs1           : 0   (157,5)
  vs0           : 0   (162,5)
* 6 immediates (16-bit each, two can be merged for 32)
  imm_5         : 0   (167,16)
  imm_4         : 0   (183,16)
  imm_3         : 0   (199,16)
  imm_2         : 0   (215,16)
  imm_1         : 0   (231,16)
  imm_0         : 0   (247,16)
* ttu? what's a ttu?
  ttu_set_btr   : 0   (263,1)
  ttu_iterate   : 0   (264,1)
  ttu_row       : 0   (265,3)
* 2 scalar slots
  s1_dest       : 0   (268,5)
  s1_y          : 0   (273,6)
  s1_x          : 0   (279,5)
  s1_op         : 0   (284,6)
  s1_pred       : 31  (290,5)
  s0_dest       : 0   (295,5)
  s0_y          : 0   (300,6)
  s0_x          : 0   (306,5)
  s0_op         : 0   (311,6)
  s0_pred       : 15  (317,5)
```

## Running a Program (WIP)

Our goal is to run a program on TPU without the driver.

```
...
openat(AT_FDCWD, "/dev/accel3", O_RDWR) = 184
mmap(NULL, 27799736, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_LOCKED, 184, 0) = 0x7f59a74b3000
# size is 0x1a830b8, aka 28MB
```

