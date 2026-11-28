# Arm control

The comma four serves the wide camera and controls three six-servo LeRobot SO followers for [Sisyphus Toss](https://github.com/commaai/xx/pull/7492). Left, middle, and right in the camera view correspond to USB serials `5CBC431994`, `5CBC431998`, and `5CBC453422`. Players send commands through the reservation gateway.

## Device installation

AGNOS supplies pyserial. Copy the Feetech SDK to the device using an administrative SSH connection:

```bash
device=comma@192.168.62.29
deps=$(mktemp -d)
uv pip install --no-deps --target "$deps" feetech-servo-sdk==1.0.0
ssh "$device" 'mkdir -p /data/arm_teleop_deps'
scp -r "$deps/scservo_sdk" "$device:/data/arm_teleop_deps/"
```

Provision `/data/arm_teleop_gateway_key` with the gateway's random service key of at least 32 characters. Put its dedicated Ed25519 SSH public key in `/data/arm_teleop_gateway.pub`. At startup the device installs the public key with forwarding restricted to loopback port 8022 and shell access disabled.

Switch the device to this branch with `tools/op.sh switch robot-arm-teleop`, then reboot. The manager starts `camerad`, `stream_encoderd`, `webrtcd`, and `arm_teleop`. The gateway uses the SSH forward for commands and camera signaling. Body’s WebRTC stack sends hardware-encoded H.264 over LAN UDP with the camera exposure time on each frame.

`ARM_TELEOP=1` keeps cameras, display, CPU cores, Panda, and USB awake, uses active cooling, and disables automatic updates. The device uses permanent MAC addresses and disables Wi-Fi power saving. Switch to a stock branch and reboot to restore normal runtime power behavior. Network profile preferences persist.

## Device API

The device listens on `127.0.0.1:8022`. Every request requires `Authorization: Bearer <service-key>`. `GET /status` returns camera status and the connection, positions, limits, faults, and control state for each follower under `arms`.

Send JSON to `POST /command` with an `arm` of `left`, `middle`, or `right`:

* `start` requires a `lease_expires_at` Unix timestamp in the next five minutes. It returns the arm's session ID, starting positions, and limits. An optional positive `speed` limits joint motion in degrees per second. The default is full servo speed.
* `target` takes six encoder positions, the current session ID, and an increasing sequence number. The device holds position when its monotonic lease expires or 400 ms pass without a command. Each arm has an independent session and watchdog.
* `stop` holds position. Omitting `arm` or passing `"arm":"all"` stops all three. Other operations default to the left arm.
* `release` turns off torque on the selected arm.

These arms use factory servo position limits. A gripper-only overload stops the squeeze while the other joints keep moving. A repeated overload releases gripper torque and can loosen the grasp. After the fault clears, command a retreat past the contact position to resume. Status reports `gripper_limited` and the raw joint faults. Other servo faults stop the arm.
