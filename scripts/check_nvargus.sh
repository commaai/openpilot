#!/bin/bash
# nvargus-daemon status check script
# Usage: ./scripts/check_nvargus.sh

echo "=== Checking nvargus-daemon process ==="
if pgrep -x nvargus-daemon > /dev/null; then
    echo "OK: nvargus-daemon is running"
    ps aux | grep nvargus-daemon | grep -v grep
else
    echo "ERROR: nvargus-daemon is not running"
    echo "  -> Run: sudo systemctl start nvargus-daemon"
    exit 1
fi

echo ""
echo "=== Testing camera session ==="
result=$(timeout 3 gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=1 ! fakesink 2>&1)

if echo "$result" | grep -q "Done Success"; then
    echo "OK: Camera session is healthy"
    exit 0
elif echo "$result" | grep -q "Failed to create CaptureSession"; then
    echo "ERROR: Camera session is in invalid state (stale session detected)"
    echo ""
    echo "To recover, run:"
    echo "  sudo systemctl restart nvargus-daemon"
    exit 1
elif echo "$result" | grep -q "No cameras available"; then
    echo "ERROR: No cameras detected"
    echo "  -> Check physical camera connection"
    exit 1
else
    echo "UNKNOWN: Unexpected state"
    echo "$result" | tail -5
    exit 1
fi
