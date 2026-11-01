// timeline seconds → camera frame number, n = slot * SEGMENT_FRAMES + index into that slot's file. A slot is a segment,
// or for a range of a route (route.range) the range's kth segment, its times shifted by (k - segment) minutes.
// With logs loaded, the camera's EncodeIdx events place frames exactly; without, frames are a fixed 20 Hz from the
// usual delay between the start of the logs and the first camera frame.
export const FPS = 20;
export const SEGMENT_FRAMES = 60 * FPS;
const FIRST_FRAME = 0.9;  // seconds, measured on recent routes (0.81-1.08)

const SERVICES = {
  qcamera: ["qRoadEncodeIdx", "qNarrowRoadEncodeIdx"],
  fcamera: ["roadEncodeIdx", "narrowRoadEncodeIdx"],
  ecamera: ["wideRoadEncodeIdx"],
  dcamera: ["driverEncodeIdx", "cabinEncodeIdx"],
};

// the position in the camera's files played back to back (slot * 60 s + frame index / FPS) of what was captured at t
export function cameraTime(store, route, camera, t) {
  const svc = store && SERVICES[camera].find((s) => store.series(`/${s}/segmentId`));
  if (svc) {
    // the last frame logged by t, then forward by its capture time: EncodeIdx is logged ~50 ms after capture
    const id = store.valueAt(`/${svc}/segmentId`, t), seg = store.valueAt(`/${svc}/segmentNum`, t);
    const k = route.range ? route.segment_numbers.indexOf(seg) : seg;
    const sof = store.valueAt(`/${svc}/timestampSof`, t) / 1e9 - store.route.t0 + (k - seg) * SEGMENT_FRAMES / FPS;
    if (id !== undefined && k >= 0) return (k * SEGMENT_FRAMES + id) / FPS + Math.max(0, t - sof);
  }
  return Math.max(0, t - FIRST_FRAME);
}

export const frameAt = (store, route, camera, t) => Math.floor(cameraTime(store, route, camera, t) * FPS + 1e-6);
