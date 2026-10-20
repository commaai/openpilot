// timeline seconds → camera frame number, n = slot * SEGMENT_FRAMES + index into that slot's file. A slot is a segment,
// or for a range of a route (route.range) the range's kth segment, its times shifted by (k - segment) minutes.
// With logs loaded, the camera's EncodeIdx events place frames exactly; without, frames are a fixed 20 Hz from 0.
export const FPS = 20;
export const SEGMENT_FRAMES = 60 * FPS;

const SERVICES = {
  fcamera: ["roadEncodeIdx", "narrowRoadEncodeIdx"],
  ecamera: ["wideRoadEncodeIdx"],
  dcamera: ["driverEncodeIdx", "cabinEncodeIdx"],
};

export function frameAt(store, route, camera, t) {
  const svc = store && SERVICES[camera].find((s) => store.series(`/${s}/segmentId`));
  if (svc) {
    // the last frame logged by t, then forward by its capture time: EncodeIdx is logged ~50 ms after capture
    const id = store.valueAt(`/${svc}/segmentId`, t), seg = store.valueAt(`/${svc}/segmentNum`, t);
    const k = route.range ? route.segment_numbers.indexOf(seg) : seg;
    const sof = store.valueAt(`/${svc}/timestampSof`, t) / 1e9 - store.route.t0 + (k - seg) * SEGMENT_FRAMES / FPS;
    if (id !== undefined && k >= 0) return k * SEGMENT_FRAMES + id + Math.max(0, Math.floor((t - sof) * FPS + 1e-6));
  }
  return Math.max(0, Math.floor(t * FPS + 1e-6));
}
