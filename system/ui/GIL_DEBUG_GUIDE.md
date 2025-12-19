# Debugging GIL (Global Interpreter Lock) Contention in UI

When the UI stutters, it's often because Python code is holding the GIL (Global Interpreter Lock) for too long, blocking the UI thread.

## Quick Start

### Method 1: Manual Timing with GILTracker (Recommended)

Add timing around suspected code blocks:

```python
from openpilot.system.ui.lib.gil_debug import GILTracker

def render(self, rect):
    # Track expensive operations
    with GILTracker("NetworkPanel.render"):
        networks = self._wifi_manager.get_networks()

    with GILTracker("NetworkPanel.text_rendering"):
        for net in networks:
            self._render_network_name(net)
```

This will automatically log when operations take longer than the threshold.

### Method 1b: Enable Background Monitoring

Enable faulthandler and basic monitoring:

```bash
# Enable faulthandler (dumps stack on SIGUSR1)
export GIL_DEBUG=1
./selfdrive/ui/ui.py

# When stutter occurs, send signal to dump stack:
# kill -USR1 <UI_PID>
```

### Method 2: Manual Timing Instrumentation

Add timing around suspected code blocks:

```python
from openpilot.system.ui.lib.gil_debug import GILTracker
import time

def render(self, rect):
    start = time.monotonic()

    # Your render code here
    with GILTracker("Widget.render"):
        self._do_expensive_operation()

    elapsed_ms = (time.monotonic() - start) * 1000
    if elapsed_ms > 10:
        print(f"Render took {elapsed_ms:.2f}ms")
```

### Method 3: Flamegraph Profiling (Best for Visual Analysis)

#### Quick Script (Easiest)

```bash
# Profile UI for 10 seconds and generate flamegraph
./system/ui/lib/profile_ui.sh 10 /tmp/ui_flamegraph.svg

# Same, but let the script find the PID using a custom `pgrep -f` pattern
./system/ui/lib/profile_ui.sh 10 /tmp/ui_flamegraph.svg "" "selfdrive/ui/ui.py"

# View the flamegraph
xdg-open /tmp/ui_flamegraph.svg
```

#### Built-in Flamegraph (Trigger with Signal)

```bash
# Enable flamegraph profiling
export GIL_FLAMEGRAPH=1
export GIL_FLAMEGRAPH_DURATION=10  # seconds
./selfdrive/ui/ui.py

# When you see stutter, trigger profiling:
kill -USR2 <UI_PID>

# Or profile automatically on startup (auto-starts)
export GIL_FLAMEGRAPH=1
export GIL_FLAMEGRAPH_DURATION=30
./selfdrive/ui/ui.py
# Flamegraph will be saved to /tmp/gil_flamegraph.svg
```

#### Manual py-spy (Most Control)

```bash
# Install py-spy
pip install py-spy

# Find UI PID
pgrep -f "selfdrive/ui/ui.py"

# Profile UI process
py-spy record -o /tmp/ui_flamegraph.svg --pid <UI_PID> --duration 30 --rate 100

# **Prove/Disprove GIL contention**
# Record *only when the GIL is held*.
# If your suspected code (e.g. PrimeState/api_get) still dominates this graph,
# that’s strong evidence it’s running while holding the GIL.
py-spy record --gil -o /tmp/ui_gil_only.svg --pid <UI_PID> --duration 30 --rate 100

# View flamegraph
xdg-open /tmp/ui_flamegraph.svg
```

#### Using faulthandler (built-in)

Add to your code:

```python
import faulthandler
import signal

# Dump stack trace on SIGUSR1
faulthandler.enable()
faulthandler.register(signal.SIGUSR1)

# Then send signal when stutter occurs:
# kill -USR1 <UI_PID>
```

### Method 4: Add Timing to Widget Renders

For network panel specifically, add timing:

```python
# In your widget's render method
def render(self, rect):
    import time
    start = time.monotonic()

    # ... existing render code ...

    elapsed_ms = (time.monotonic() - start) * 1000
    if elapsed_ms > 5:  # Log slow renders
        print(f"[{self.__class__.__name__}] Render took {elapsed_ms:.2f}ms")
```

## Where Logs Are Saved

**By default:**
- **On comma device**: Logs go to `cloudlog` → `/data/log/` directory (rotating log files)
- **On PC**: Logs go to `stderr` (console/terminal)

**To specify a file:**
```bash
export GIL_LOG_FILE=/tmp/gil_debug.log  # Logs to specific file
```

**To disable cloudlog and use stderr:**
```bash
export GIL_USE_CLOUDLOG=0  # Use stderr instead
```

**View logs on comma device:**
```bash
# View recent cloudlog entries
tail -f /data/log/*.log | grep GIL

# Or use the log viewer tool
```

## Common Culprits

1. **Heavy string operations**: String formatting, regex, text rendering
2. **File I/O**: Reading/writing files, network requests
3. **List/dict operations**: Large list comprehensions, dict lookups
4. **Import statements**: Dynamic imports in render loop
5. **Message parsing**: Cap'n Proto message parsing
6. **Widget layout calculations**: Complex layout math

## Example: Finding Network Panel Stutters

```python
# In network panel widget
from openpilot.system.ui.lib.gil_debug import GILTracker

def render(self, rect):
    with GILTracker("NetworkPanel.render"):
        # Network scanning
        with GILTracker("NetworkPanel.scan"):
            networks = self._wifi_manager.get_networks()

        # Text rendering
        with GILTracker("NetworkPanel.text"):
            for net in networks:
                self._render_network_name(net)
```

## Analyzing Results

When GIL_DEBUG is enabled, check:

1. **Stderr/logs**: Look for stack traces showing which functions hold GIL
2. **Statistics**: Call `print_stats()` to see cumulative stats
3. **Patterns**: Look for repeated offenders in the stats

Example output:
```
================================================================================
GIL HELD FOR 15.23ms (> 10ms threshold)
================================================================================
Stack trace:
  File "system/ui/widgets/label.py", line 234, in render
    text = self._format_text()
  File "system/ui/widgets/label.py", line 456, in _format_text
    return self._text.replace('\n', ' ')
================================================================================
```

## Tips

- Start with `GIL_THRESHOLD_MS=5` to catch smaller stutters
- Use `GIL_LOG_FILE` to avoid cluttering stderr
- Combine with `SHOW_FPS=1` to correlate GIL holds with FPS drops
- For production debugging, use `py-spy` as it has minimal overhead

