# SQTT Profiling

## Getting SQ Thread Trace

Only supported on 7900XTX, requires either AM (`rmmod amdgpu`) or disabling power gating on AMD (`ppfeaturemask=0xffff3fff`, don't forget to rebuild initramfs)

SQTT is implemented on top of normal tinygrad PROFILE=1, `PROFILE=1 SQTT=1` to get profile pickle with sqtt data embedded in it.

`SQTT_BUFFER_SIZE=X` to change size of SQTT buffer (per shader engine, 6 SEs on 7900xtx) in megabytes, default 256.

`SQTT_ITRACE_SE_MASK=X` to select for which shader engines instruction tracing will be enabled, -1 is all, 0 is none (instruction tracing disabled), >0 is
bitfield/mask for SEs to enable instruction tracing on. Masking shader engines will give smaller file sizes at a cost of less hits and kernels that
don't have any wavefront on first simd of shdaer engine with instruction tracing enabled will not have instruction timings.
The default is 2 (second shader engine only), only one for file size reasons, second instead of first because dispatch starts from it so there is
greater chance that kernels with small global size will have instruction tracing data.
 
Note that instruction tracing might not be available for kernels with small global dims, this is not a bug, but it can be improved with various hacks
to the point where it can reliably trace a kernel consisting of a single wavefront (am only, not quite reliable under amdgpu due to waves sometimes
being dispatched starting from different simds). More info in comments in ops_amd.py

## Converting pickled profile with SQTT data into RGP file

```bash
extra/sqtt/rgptool.py create "/tmp/profile.pkl.$USER" -o /tmp/gpu0.rgp
```

Then load gpu0.rgp into Radeon GPU Profiler. It works just fine both in wine (macos, native version available for linux) and via ssh X forwarding

If multiple gpus are used you can select which one to export with `-d` like this:

```bash
extra/sqtt/rgptool.py create "/tmp/profile.pkl.$USER" -d 'AMD:5' -o /tmp/gpu5.rgp
```
