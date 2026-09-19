# Experimental AGNOS early UI profile

This branch accompanies `agnos-builder`'s `feature/fast-userspace-ui` branch. It was tested on a comma four with native font loading and an already configured openpilot installation. Settings is constructed eagerly: its first tap must not pay an import/construction penalty.

## Integration contract

- The ordinary launcher still owns registration, updates, build checks and manager startup. `/data/openpilot/prebuilt` skips the build; it is a device-local flag, not a tracked file.
- `/data/openpilot/.agnos-early-ui` explicitly opts a fixed, built checkout into the matching AGNOS `openpilot-early-ui.service`. Do not create it for unpatched installations. Remove it before switching to an unrelated branch or updating the checkout/OS.
- AGNOS launches this checkout's UI on CPU 5 before background services. The manager's `EarlyUIProcess` adapter adopts its PID via `/run/openpilot-early-ui.pid`; systemd owns restart and reaping.
- After the actual first `rl.end_drawing()`, the UI writes monotonic seconds to `/tmp/boot-first-frame-PID`. AGNOS waits at most eight seconds for this diagnostic readiness signal and then releases background work. `/tmp` is a new tmpfs on every boot.
- `AGNOS_BOOT_UI_PRIORITY=1` makes manager wait briefly for a UI message before starting other processes. The OS launcher also uses a bounded first-frame wait.
- `/tmp/boot-stages` retains phase instrumentation so the result can be reproduced. First-frame time minus `systemd-analyze time`'s kernel time is the userspace metric. Systemd target completion is a different measurement.

## Retained UI changes

Only the selected screen layout is imported. Completed onboarding no longer constructs the terms/training UI. Hidden driving views are prepared just after the first frame (or when first rendered). Prime HTTP/auth imports run in their worker after the first frame. Native font loading, eager settings and the original NumPy/driving-alert code remain intact.

## Validation and limits

The user confirmed normal fonts and prompt first-tap Settings after the final rollback. The UI remained active without automatic restarts and systemd reported no failed units. The earlier configuration reached 2.974 and 2.980 seconds userspace; settings was still lazy at those measurements. Sub-two-second boot was not achieved, and the user ended that optimization effort in favor of responsiveness. See the builder's experiment record for the final reboot result.

This is a development profile, not a release certification. Driving transitions, first installation, factory-reset interaction, checkout swaps and every manager shutdown/restart path need full integration coverage before enabling it by default. In particular, the service/manager adapter and diagnostic readiness protocol are experimental. No font atlas cache or NumPy/driving-alert import experiment is included.

Final device reboot of the committed configuration: **3.333725 seconds userspace to first UI frame** (`6.285725 - 2.952`). No failed systemd units; UI automatic restart count zero; CPU cap restored to 1689600 kHz.
