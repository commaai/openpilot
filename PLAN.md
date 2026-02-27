# Plan: Process all 3 cameras on IFE with 2 IFE hardware instances

## Goal

Process all 3 cameras on the IFE, but we only have 2x full IFEs. We should have enough time to process images from two different cams in one "cycle", but we just need to configure the ISP/IFE to do this. Stagger their strobe and pass them through the IFE at different times.

Test on a real device by SSHing with: `tools/op.sh adb`

## Context

3 cameras (wide, road, driver), 2 full IFE hardware instances on Titan 170 ISP (SDM845). Currently wide uses BPS processing (`ISP_BPS_PROCESSED`) to free up an IFE. We want all 3 cameras processed through the IFE pixel pipeline (`ISP_IFE_PROCESSED`). The IFE pixel pipeline takes ~1ms per frame, and the frame period is 50ms at 20fps.

**Constraints identified during investigation:**
- One IFE has one CSID input bound to one PHY — cannot acquire one IFE with 2 PHY inputs
- One ISP device handle cannot participate in 2 separate links
- CSID PHY routing is configured at acquire time, but the `PHY_NUM_SEL` register (0x4100, bits [21:20]) is writable via CDM at runtime

**Solution:** Stagger sensor frame timing via FSIN delay registers, then software-switch `CSID_CSI2_RX_CFG0.PHY_NUM_SEL` between frames so one IFE processes two cameras sequentially within each 50ms cycle.

## Architecture

```
IFE_0 (shared, time-multiplexed):
  Session A: ISP(PHY_0) + wide_sensor → link_A
  Wide:   PHY_0, SOF at t=0,    readout t=0..15ms,   IFE processing t=15..16ms
  Driver: PHY_2, SOF at t=18ms, readout t=18..33ms,  IFE processing t=33..34ms
  PHY_NUM_SEL switches: 0→2 at t=16ms, 2→0 at t=34ms (software-driven)

IFE_1 (dedicated, unchanged):
  Session B: ISP(PHY_1) + road_sensor → link_B
  Road:   PHY_1, normal flow
```

The driver camera does NOT acquire its own ISP device. It has its own session (for sensor/CSIPHY management) but shares IFE_0 by having its frames processed in the wide camera's event handler, using direct `device_config` calls with `OpcodesIFEInitialConfig`.

## Timing

At 20fps, frame period = 50ms. Sensor readout times:
- OX03C10: ~14.7ms
- OS04C10: ~11ms

```
t=0ms     Wide SOF (PHY_0) ─── IFE_0 processes wide frame ───┐
t=~15ms   Wide done. Sync fence signals.                      │
t=~15ms   Software: PHY_NUM_SEL=2 (switch to PHY_2)           │ 50ms cycle
t=~18ms   Driver SOF (PHY_2) ─── IFE_0 processes driver ───┐  │
t=~33ms   Driver done. Sync fence signals.                  │  │
t=~33ms   Software: PHY_NUM_SEL=0 (switch back to PHY_0)    │  │
t=~33ms   Schedule next wide request (enqueue_frame)         │  │
t=50ms    Next cycle ────────────────────────────────────────┘──┘
```

Wide+driver total: ~34ms of the 50ms budget. 16ms margin.

## Per-frame flow (detailed)

1. **Wide SOF arrives** (V4L2 event, `request_id=N`). Request manager applies request N to IFE_0.
2. **Wide frame processes** on IFE_0 (CDM already configured for wide pipeline + wide output buffer).
3. **Wide sync fence signals** (~15ms after SOF). `handle_camera_event` processes wide frame.
4. **Switch PHY for driver**: Submit `device_config(OpcodesIFEInitialConfig)` to wide's ISP device with CDM containing `write_cont(0x4100, {0x232103})` (PHY_NUM_SEL=2). This executes immediately (init configs bypass request manager).
5. **Submit driver frame config**: Another `device_config(OpcodesIFEInitialConfig)` with driver's output buffer (io_cfg) + driver's sync fence + minimal CDM (build_update for driver).
6. **Driver SOF arrives** on PHY_2 (~18ms into cycle). IFE_0 starts processing driver data. The SOF generates a V4L2 event with `request_id=0` (no scheduled request) — harmlessly ignored by wide's `handle_camera_event`.
7. **Driver sync fence signals** (~33ms). Process driver frame, call driver's `sendState`.
8. **Switch PHY back**: Submit `device_config(OpcodesIFEInitialConfig)` with `write_cont(0x4100, {0x032103})` (PHY_NUM_SEL=0).
9. **Enqueue next wide frame**: `enqueue_frame(N + ife_buf_depth)` schedules wide's next request. This is delayed to AFTER driver completes, so the driver's SOF (#6) doesn't trigger a premature request application.

## CSID register details

`CSID_CSI2_RX_CFG0` at offset **0x4100**, fields:
```
[1:0]   NUM_ACTIVE_LANES = 3  (4 lanes)
[5:4]   DL0_INPUT_SEL    = 0
[9:8]   DL1_INPUT_SEL    = 1
[13:12] DL2_INPUT_SEL    = 2
[17:16] DL3_INPUT_SEL    = 3
[21:20] PHY_NUM_SEL      = variable (0 for PHY_0, 2 for PHY_2)
[24]    PHY_TYPE_SEL     = 0  (D-PHY)
```

Full register values (derived from lane_cfg=0x3210, 4-lane D-PHY):
- **PHY_0 (wide):**  `0x032103`
- **PHY_2 (driver):** `0x232103`

## FSIN delay configuration

Both sensors use FSIN (external frame sync) mode. We need the driver sensor to start its frame ~18ms after wide. The delay is set via sensor i2c registers during init.

**OX03C10:**
- Delay registers: `0x3836` (high), `0x3837` (low)
- Current wide values: `0x1F`, `0x40`
- Driver needs larger delay — calculate based on line_time = readout_time_ns / VTS

**OS04C10:**
- Delay registers: `0x382a` (high), `0x382b` (low)
- Current values: `0x00`, `0x0c` (12 lines, ~78µs — essentially no stagger)
- Driver needs ~18ms delay: 18ms / (11ms / 1692 lines) ≈ 2769 lines = `0x0AD1`

**Note:** Exact delay values need experimental tuning. The key requirement is that the driver's MIPI data arrives AFTER wide's frame is fully processed (>16ms gap from wide SOF).

## Files to modify

### 1. `system/camerad/cameras/hw.h`
- Change `WIDE_ROAD_CAMERA_CONFIG.output_type` from `ISP_BPS_PROCESSED` to `ISP_IFE_PROCESSED`
- Add `int ife_partner` to `CameraConfig` (-1 for none, camera_num of primary for secondary)
  - Wide: `ife_partner = -1` (primary)
  - Road: `ife_partner = -1` (solo)
  - Driver: `ife_partner = 0` (shares IFE with wide)

### 2. `system/camerad/cameras/spectra.h`
- Add to `SpectraCamera`:
  ```cpp
  SpectraCamera *ife_partner = nullptr;   // partner camera sharing this IFE
  bool is_ife_secondary() const;          // true if this camera is the secondary on a shared IFE
  void config_ife_secondary(int idx);     // process a frame for the secondary camera
  ```
- Add driver sync fence storage for secondary frames

### 3. `system/camerad/cameras/spectra.cc`

#### `openSensor()` — driver camera
- Driver still creates its own session (for sensor/CSIPHY device management)
- Sensor probe, acquire, and i2c init are unchanged
- **NEW**: After init, send additional i2c write to set the FSIN delay registers for stagger:
  ```cpp
  if (cc.ife_partner >= 0) {
    // Increase FSIN delay so this sensor fires ~18ms after the primary
    struct i2c_random_wr_payload fsin_delay[] = {
      {0x382a, 0x0A}, {0x382b, 0xD1},  // OS04C10: ~18ms delay
    };
    sensors_i2c(fsin_delay, std::size(fsin_delay), CAM_SENSOR_PACKET_OPCODE_SENSOR_CONFIG, sensor->data_word);
  }
  ```

#### `configISP()` — driver camera
- **Skip ISP device acquisition entirely** (no `device_acquire(m->isp_fd, ...)`)
- **Skip IFE buffer allocation** (`ife_cmd`, LUTs, etc.)
  ```cpp
  if (is_ife_secondary()) {
    // ISP device is owned by the primary camera (ife_partner)
    // We'll use their isp_dev_handle and session for device_config calls
    return;
  }
  ```

#### `configICP()` — driver camera
- **Skip entirely** (no BPS needed since we're using IFE)

#### `camera_map_bufs()` — driver camera
- Map driver's YUV output buffers into the ISP IOMMU using `m->device_iommu`
- Same code as current, just ensure `icp_dev_handle` check is skipped

#### `linkDevices()` — driver camera
- **Skip ISP link creation** (driver doesn't have its own ISP device)
- Still start CSIPHY:
  ```cpp
  if (is_ife_secondary()) {
    // No link for secondary — frames are processed via primary's event handler
    // But still start CSIPHY so MIPI data flows
    ret = device_control(csiphy_fd, CAM_START_DEV, session_handle, csiphy_dev_handle);
    return;
  }
  ```

#### `handle_camera_event()` — wide camera (primary)
- After processing wide's own frame and BEFORE `enqueue_frame`:
  ```cpp
  // Process partner (driver) frame on the shared IFE
  if (ife_partner != nullptr) {
    ife_partner->config_ife_secondary(/* buf_idx */);
  }
  // Now safe to enqueue next wide frame
  enqueue_frame(request_id + ife_buf_depth);
  ```

#### NEW: `config_ife_secondary(int idx)` — processes one driver frame
This is the core new function. Steps:
1. **Switch PHY**: Build a `cam_packet` with `OpcodesIFEInitialConfig`, CDM containing `write_cont(0x4100, {0x232103})`. Submit via `device_config(m->isp_fd, ife_partner->session_handle, ife_partner->isp_dev_handle, ...)`.
2. **Submit driver frame**: Build another `cam_packet` with:
   - `OpcodesIFEInitialConfig` (for immediate execution)
   - CDM: `build_update()` for driver camera config (identical to wide since same sensor, same vignetting=false)
   - `io_cfg`: driver's `buf_handle_yuv[idx]`, NV12 format, FULL output, driver's sync fence
   - Submit via `device_config` using wide's ISP handles
3. **Wait for frame**: `CAM_SYNC_WAIT` on driver's sync fence (timeout ~50ms). The driver sensor's MIPI data arrives after the FSIN delay, and IFE_0 processes it.
4. **Process frame**: Fill driver's `buf.cur_frame_data` with frame metadata.
5. **Switch PHY back**: Submit `device_config` with CDM `write_cont(0x4100, {0x032103})` (PHY_NUM_SEL=0).
6. **Destroy sync fence** and return.

#### `camera_close()` — driver camera
- Release sensor and CSIPHY normally (in driver's own session)
- Do NOT release ISP device (owned by wide)
- Do NOT unlink (driver has no link)
- Destroy driver's own session

### 4. `system/camerad/cameras/camera_qcom2.cc`

#### Initialization
```cpp
// After creating CameraState objects, link IFE partners
cams[0]->camera.ife_partner = &cams[2]->camera;  // wide's partner is driver
```

#### Initialization order
- Wide must be initialized BEFORE driver (so ISP device is acquired first)
- Current order (0, 1, 2) already satisfies this

#### Event dispatch
- Driver SOF events will appear with wide's `session_handle` and `request_id=0`
- These are already handled: `validateEvent` returns false for `request_id=0`, so they're silently ignored
- After wide processes its frame, it triggers driver processing internally

#### Driver sendState
- After `config_ife_secondary` returns, call driver's `sendState`:
  ```cpp
  // In the poll loop, after wide's handle_camera_event returns true:
  if (cam->camera.ife_partner != nullptr) {
    cams[cam->camera.cc.ife_partner]->sendState();  // wrong direction, see below
  }
  ```
  Actually, the partner pointer is on the primary (wide) pointing to secondary (driver). So:
  ```cpp
  for (auto &cam : cams) {
    if (event_data->session_hdl == cam->camera.session_handle) {
      if (cam->camera.handle_camera_event(event_data)) {
        cam->sendState();
        // If this camera has a partner, also send the partner's state
        if (cam->camera.ife_partner != nullptr) {
          // Find the partner CameraState and sendState
          for (auto &partner : cams) {
            if (&partner->camera == cam->camera.ife_partner) {
              partner->sendState();
              break;
            }
          }
        }
      }
      break;
    }
  }
  ```

## Simplifications

- **Same sensor type on all cameras**: IFE pipeline config (gamma, linearization, color correction, black level) is identical for wide and driver. `build_update()` produces identical CDM for both.
- **Both wide and driver have `vignetting_correction = false`**: No register reconfiguration needed between frames. Only PHY_NUM_SEL and output buffer change.

## Key risks and mitigations

### Risk 1: `OpcodesIFEInitialConfig` with `io_cfg` might not work
The kernel might not process `io_cfg` (output buffer + sync fence) for init-opcode packets. In that case:
- **Fallback A**: Try `OpcodesIFEUpdate` without `CAM_REQ_MGR_SCHED_REQ`. The kernel might process it immediately.
- **Fallback B**: Write the output buffer address directly via CDM register writes to the IFE WR engine registers, and manage sync fences manually.

### Risk 2: PHY_NUM_SEL switch via CDM might not take effect immediately
The CSID might buffer the PHY selection and only apply it at certain boundaries.
- **Mitigation**: Switch PHY during the inter-frame gap (after wide's readout completes, before driver's SOF). The 3ms gap should be sufficient.

### Risk 3: Driver SOF events confuse the request manager
When PHY_NUM_SEL=2 and the CSID sees driver's SOF, it generates a SOF IRQ. The request manager dispatches this as a V4L2 event on wide's link with `request_id=0` (no scheduled request).
- **Mitigation**: Already handled — `validateEvent` returns false for `request_id=0`.
- **Watch out**: Ensure `invalid_request_count` doesn't accumulate from these benign events and trigger a `clearAndRequeue`.

### Risk 4: Blocking wide's event handler for ~18ms delays road camera processing
The single-threaded poll loop blocks while waiting for the driver frame. Road SOF events queue up.
- **Mitigation**: Road's IFE_1 processes in parallel. By the time we get to road's event, its sync fence has already signaled. `waitForSync` returns immediately. Total cycle: ~34ms wide+driver + ~0ms road = 34ms.

## Verification

1. Build: `scons system/camerad/`
2. SSH to device: `tools/op.sh adb`
   there is also a nice helper script in system/camerad/ that gets great logging/debug info out of the kernel and ISP/BPS
3. Run with `DEBUG_FRAMES=1` to verify:
   - Wide SOFs arrive with valid request_ids
   - Driver SOFs appear ~18ms after wide SOFs (with request_id=0)
   - Road SOFs are unaffected
4. Verify all 3 camera streams produce valid NV12 frames in VisionIPC
5. Check timing: `processing_time` for all cameras should be reasonable
6. Monitor for sync failures, frame drops, or `clearAndRequeue` events
7. Verify no CSID errors in dmesg (corrupt frames from mid-frame PHY switch)
