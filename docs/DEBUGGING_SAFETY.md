# Debugging Panda Safety with Replay Drive + LLDB

By default, `libsafety.so` is built with optimizations and stripped symbols, which makes C-level debugging impractical.
This workflow fixes that by:

1. Building `libsafety` with debug symbols and no optimizations.
2. Running `replay_drive.py` with a user-provided route.
3. Attaching LLDB automatically from VS Code.

---

## Building with Debug Symbols

Rebuild `libsafety` with the `--unoptimized` flag:

```bash
scons -u -j$(nproc) --unoptimized
```

This enables the debug flags:

* `-g` (include symbols)
* `-O0` (disable optimizations)
* `-fno-omit-frame-pointer` (keep stack traces usable)

---

## Debugging Workflow

1. **Build with debug symbols**

   ```bash
   scons -u -j$(nproc) --unoptimized
   ```

2. **Start the debugger in VS Code**

    * Select **Replay drive + Safety LLDB**.
    * Enter the route or segment when prompted.

3. **Attach LLDB**

    * When prompted, pick the running **`replay_drive` process**.
    * ⚠️ Attach quickly, or `replay_drive` will start consuming messages.

   ✅ Tip: Add a Python breakpoint at the start of `replay_drive.py` to pause execution and give yourself time to attach LLDB.

4. **Set breakpoints in VS Code**
   Breakpoints can be set directly in `libsafety.c` (or any C file).
   No extra LLDB commands are required — just place breakpoints in the editor.

5. **Resume execution**
   Once attached, you can step through both Python and C safety code as CAN logs are replayed.

---

## Notes

* Always rebuild with `--unoptimized` before debugging.
* Use short routes for quicker iteration.
* Pause `replay_drive` early to avoid wasting log messages.

### Video

View a demo of this workflow on the PR that added it: https://github.com/commaai/openpilot/pull/36055#issue-3352911578