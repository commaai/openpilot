# Arm control

Wide-camera video from a comma four and leader/follower control for a six-servo LeRobot SO arm. The device runs the `robot-arm-teleop` openpilot branch. The workstation serves a local UI and reads the USB leader; video and commands travel through SSH.

This setup uses leader `5CBC431223` on the workstation and three followers on the device: left `5CBC431994`, middle `5CBC431998`, and right `5CBC453422`. Names refer to the wide-camera view. USB serial numbers are used instead of changing `/dev/ttyACM` numbers.

## Workstation

```bash
tools/arm_teleop/run.sh --device 192.168.63.237 --key ~/xx/private/key/id_rsa
```

Open http://127.0.0.1:8070. `uv` installs the small Python environment on the first run. The SSH host key must already be trusted. `--leader`, `--follower left|middle|right`, `--port`, and `--tunnel-port` override the defaults.

1. Wait for live video and both arm connections.
2. Select the follower arm, movement scale, and maximum speed. Start with 0.5× and 30°/s.
3. Choose **Match leader joint positions** for 1:1 encoder tracking or **Relative movement** to apply scaled changes from the two starting poses. Click **Start following**.
4. **Pause / hold position**, Space, or Escape stops following and holds the follower's current position. Pause before selecting another arm. Restarting relative control resets its reference poses.
5. **Release follower torque** makes the follower free to move by hand. Support it before releasing.

Changing tabs, leaving the browser window, closing the page, losing the camera, or losing the command connection stops following. Reconnection never resumes motion automatically. The device independently expires commands after 400 ms. Speed and position limits are enforced on the device, and encoder jumps are rejected on the workstation.

These arms currently have factory position limits, not calibrated mechanical joint limits. Pose matching targets the same encoder readings. Physical alignment depends on the servo horns and encoder zeros on each arm. It preserves the existing EEPROM configuration. Per-joint Reverse controls accommodate an opposite assembly direction; change them while paused. For full calibrated pose matching, first perform the [LeRobot arm calibration](https://huggingface.co/docs/lerobot/so101).

## Device installation

The device needs `scservo_sdk` from `feetech-servo-sdk==1.0.0` in `/data/arm_teleop_deps`. AGNOS already supplies pyserial, NumPy, and Pillow. From a prepared workstation environment:

```bash
ssh -i ~/xx/private/key/id_rsa comma@192.168.63.237 'mkdir -p /data/arm_teleop_deps'
scp -r -i ~/xx/private/key/id_rsa tools/arm_teleop/.venv/lib/python*/site-packages/scservo_sdk \
  comma@192.168.63.237:/data/arm_teleop_deps/
```

Switch the device to this branch with `tools/op.sh switch robot-arm-teleop`, then reboot. Openpilot's manager starts `camerad`, `arm_teleop`, and `arm_power`; the control service listens only on device loopback port 8022. The workstation also listens only on loopback. No cloud hosting is involved.

`launch_env.sh` enables `ARM_TELEOP=1` on this branch. This keeps the cameras, display, CPU cores, performance governors, and Panda awake; uses the active cooling curve; blocks openpilot shutdown requests; masks AGNOS idle power monitors and OS sleep targets; and disables USB autosuspend, including hotplugged hubs. NetworkManager profiles use permanent MAC addresses and disable Wi-Fi power saving and scan randomization. Wi-Fi changes apply when the connection next activates, so reboot after installation. Hardware thermal protections remain active; software cannot prevent loss of physical power.

Automatic openpilot updates are disabled in this mode. Switch to a stock branch and reboot to restore stock runtime power behavior. Network profile preferences for permanent MAC addresses and disabled Wi-Fi power saving persist.

## Development

`device.py` owns one bus, control session, and command watchdog per follower, plus the shared MJPEG encoder. `workstation.py` owns the leader bus, SSH tunnel, and browser session. `servos.py` is a small wrapper around the Feetech SDK with read-only discovery. `power.py` applies the device runtime settings and maintains USB power/permissions after reconnects.

Neither server starts motion on its own. Probe and status reads do not enable torque. Motion commands are accepted only after an explicit Start and require the current session ID and increasing sequence numbers.

## Reserved game sessions

The companion [xx service](https://github.com/commaai/xx/pull/7492) books five-minute slots at `https://sisyphustoss.comma.life` and issues a different API token for each reservation. Save its token file, stop any older workstation teleop service, and run:

```bash
python tools/arm_teleop/workstation.py --game-url https://sisyphustoss.comma.life \
  --reservation <reservation-id> --token-file /path/to/balltoss-token.txt
```

Open `http://127.0.0.1:8070/play` for the reservation controller. Choose the left, middle, or right follower before or during the slot. When the slot opens, click **Start following** to move the follower toward the leader’s current encoder positions at up to 30°/s and then follow 1:1. The page shows the reservation window, control state and live camera. Pause before switching arms. The gateway ends control at the slot deadline, and the workstation releases the USB leader. The full joint and speed controls remain at `http://127.0.0.1:8070/`.

On the device, `/data/arm_teleop_gateway_key` enables managed mode. It must contain a random service key of at least 32 characters and match the gateway's credential. Every device HTTP request then requires `Authorization: Bearer <service-key>`, and `start` requires a `lease_expires_at` Unix timestamp no more than five minutes ahead. The device converts that timestamp once to a monotonic deadline and holds the arm when the lease ends. The 400 ms command watchdog remains active. Keep device port 8022 on loopback, and restart the device manager after installing the key.

Without the key file, the original direct workstation mode is available. Do not remove this file from a device assigned to the reservation service.

Place the gateway's dedicated Ed25519 SSH public key at `/data/arm_teleop_gateway.pub`. At teleop startup, the power setup installs it into AGNOS's `/tmp/authorized_keys` with a restricted forwarding destination and a disabled shell. Existing administrative SSH keys are preserved. The private key stays on the gateway host.

During a session, an overload confined to the gripper stops the squeeze while the other five joints keep following. The controller resets the gripper target to its measured position and switches off gripper torque if the overload is reported again. This can loosen the grasp. Move the grip back past the contact position to resume once the servo clears its fault. The controller shows the restriction and the device reports `gripper_limited` alongside its raw `faults`. Other servo faults still stop the arm.

## Multi-arm API

`GET /status` returns `arms.left`, `arms.middle`, and `arms.right`, each with its serial, connection state, positions, limits, faults, and control state. The legacy `arm` field remains an alias for the left arm.

Include `"arm":"left"`, `"arm":"middle"`, or `"arm":"right"` on `start`, `target`, `stop`, and `release`. Start each arm separately and stream targets with that arm's session and increasing sequence numbers. Sessions and the 400 ms watchdog are independent; an expired session cannot command another arm.

Omitting `arm` selects the left arm, except for `stop`, which stops all arms. `{"op":"stop","arm":"all"}` also stops all arms. Torque release always applies to one arm. In managed mode all three arms use the same reservation deadline; expiry, cancellation, and gateway shutdown stop all three.

The device disables TCP delayed packet coalescing for HTTP responses and resizes YUV planes before RGB conversion to reduce encoding time. `/frame` returns the latest JPEG with `X-Capture-Time` (exposure start in Unix seconds) and `X-Frame-Id`; `/video` includes the capture timestamp on each MJPEG part. Direct workstation mode uses independent SSH connections for controls and video; `--video-tunnel-port` defaults to 18023. The reservation gateway pulls one fresh frame at a time and discards stale frames.
