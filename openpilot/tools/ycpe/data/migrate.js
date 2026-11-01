// column-level port of selfdrive/test/process_replay/migration.py, plus FIXES for schema moves it has no migration
// for. It rewrites one segment's reader output (dense columns, before compact()) into the current schema in place,
// with aliased and derived columns and no capnp re-encoding.
//
// Route context: migrations that need initData or carParams read them from the segment if it has them, otherwise
// from routeContext, which the loader fills from the context migrate() returns for an earlier segment. Python
// migrates each log on its own, so the parity test passes {}.
//
// Some migrations read lists longer than capnp.js's per-index limit, so the reader keeps them (withKeep) and
// migrate() drops them again afterwards. When the reader skipped one anyway (svc.lists), the derived paths are
// skipped and reported in `missing`.
import schema from "./schema.json" with {type: "json"};
import tables from "./migrate.json" with {type: "json"};
import {pattern} from "./capnp.js";

const STRUCTS = Object.fromEntries(schema.structs.map((s) => [s.name, s]));
const SCALAR = new Set(["b", "i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64", "f32", "f64", "e"]);
const WIDE = new Set(["f64", "i64", "u64", "u32", "i32"]);
const MAX_LIST = 16;  // capnp.js
const T_IDXS = Array.from({length: 33}, (_, i) => 10 * (i / 32) ** 2);  // ModelConstants.T_IDXS

// the long lists the migrations read
const KEEP = {
  sensorEventsDEPRECATED: [""],
  longitudinalPlan: ["/speeds", "/accels"],
  modelV2: ["/position/x", "/position/y", "/position/z", "/laneLines/*/y"],
};

const at = (col, i) => col?.v[i];
const dropped = (svc, path) => svc.lists?.has(path);  // a long list the reader skipped (not in KEEP)
const has = (col, i) => { const x = at(col, i); return x !== undefined && x !== null && x === x; };
const field = (st, name) => st.fields.find((f) => f[0] === name);
const names = (st, name) => schema.enums[field(st, name)[4]];
const ns = (svc) => svc.t.length;

function column(type, n, fn, enumNames) {
  const v = type === "t" ? new Array(n) : new (WIDE.has(type) ? Float64Array : Float32Array)(n);
  for (let i = 0; i < n; i++) v[i] = fn(i) ?? (type === "t" ? null : NaN);
  return enumNames ? {type, names: enumNames, v} : {type, v};
}

const ones = (n) => new Uint8Array(n).fill(1);

function pick(arr, idx) {
  const v = new arr.constructor(idx.length);
  for (let k = 0; k < idx.length; k++) v[k] = arr[idx[k]];
  return v;
}

// per event: does any column at or under prefix have a value (i.e. does the struct or list item exist)
function present(cols, prefix, n) {
  const ps = Object.keys(cols).filter((p) => p === prefix || p.startsWith(prefix + "/"));
  return Uint8Array.from({length: n}, (_, i) => ps.some((p) => has(cols[p], i)));
}

// one field of a new message: a source column, a function of the event index, or a constant
function value(v, type, n, enumNames) {
  if (typeof v === "function") return column(type, n, v, enumNames);
  if (typeof v !== "object") return column(type, n, () => v, enumNames);
  if (v.type !== type) return column(type, n, (i) => at(v, i), enumNames);
  return enumNames ? {...v, names: enumNames} : v;
}

// the columns of a newly built struct at prefix: fields from `set` (keyed by path within the struct; null leaves one
// unset), defaults for the other scalars (groups included, union members and pointers left unset).
// mask: the events that have the struct.
function struct(st, prefix, n, set = {}, mask = null, rel = "", cols = {}) {
  for (const [name, disc, type, , ref, def] of st.fields) {
    const sub = `${rel}/${name}`;
    const enumNames = type === "e" ? schema.enums[ref] : undefined;
    if (type === "g") struct(schema.structs[ref], prefix, n, set, mask, sub, cols);
    else if (set[sub] === null) continue;
    else if (set[sub] !== undefined) cols[prefix + sub] = value(set[sub], type, n, enumNames);
    else if (SCALAR.has(type) && disc < 0) {
      const d = type === "b" && def ? 1 : 0;  // no other non-zero defaults in the structs built here
      cols[prefix + sub] = value(mask ? (i) => mask[i] ? d : NaN : d, type, n, enumNames);
    }
  }
  return cols;
}

// initData and carParams facts the migrations need, from the first message of each
export function context(services) {
  const ctx = {};
  const init = services.initData?.cols;
  if (init) ctx.deviceType = at(init["/deviceType"], 0) ?? 0;
  const cp = services.carParams?.cols;
  if (cp) ctx.carFingerprint = at(cp["/carFingerprint"], 0) ?? "";
  return ctx;
}

const SENSORS = {acceleration: "accelerometer", gyro: "gyroscope", gyroUncalibrated: "gyroscope", light: "lightSensor", proximity: "lightSensor",
                 magnetic: "magnetometer", magneticUncalibrated: "magnetometer", temperature: "temperatureSensor"};

function sensorEvents(s, ctx, missing) {
  const src = s.sensorEventsDEPRECATED, n = ns(src), st = STRUCTS.SensorEventData;
  if (dropped(src, "")) missing.push("sensorEventsDEPRECATED");
  const members = st.fields.filter((f) => f[1] >= 0).map((f) => f[0]), first = st.fields.find((f) => f[1] === 0)[0];
  const paths = Object.keys(src.cols);
  // the union's active member is the one with values (the reader writes no others), else a null first member.
  // A member's shallowest column (a scalar, or a scalar of its struct) has a value whenever it's active.
  const depth = (p) => p.split("/").length;
  const items = [];
  for (let j = 0; paths.some((p) => p.startsWith(`/${j}/`)); j++) {
    items.push(members.map((m) => {
      const ps = paths.filter((p) => p === `/${j}/${m}` || p.startsWith(`/${j}/${m}/`)).sort((a, b) => depth(a) - depth(b));
      return [m, src.cols[ps[0]]];
    }).filter(([, col]) => col));
  }
  const out = {};  // service: [[event, item, null member]]
  for (let i = 0; i < n; i++) {
    items.forEach((byMember, j) => {
      if (!has(src.cols[`/${j}/timestamp`], i)) return;
      const m = byMember.find(([, col]) => has(col, i))?.[0];
      if (SENSORS[m ?? first]) (out[SENSORS[m ?? first]] ||= []).push([i, j, !m]);
    });
  }
  // source, timestamp and the service's union members are copied; the deprecated group is left at defaults
  const rels = new Map();
  for (const p of paths) rels.set(p.replace(/^\/\d+/, ""), src.cols[p]);
  for (const [name, entries] of Object.entries(out)) {
    const cols = struct(st, "", entries.length);
    for (const [rel, {type, names}] of rels) {
      if (rel !== "/source" && rel !== "/timestamp" && SENSORS[rel.split("/")[1]] !== name) continue;
      const byItem = items.map((_, j) => src.cols[`/${j}${rel}`]);
      cols[rel] = column(type, entries.length, (k) => at(byItem[entries[k][1]], entries[k][0]), names);
    }
    if (SENSORS[first] === name && entries.some((e) => e[2])) {  // a null member copies as a struct of defaults
      const defaults = struct(schema.structs[field(st, first)[4]], `/${first}`, entries.length);
      for (const [p, {type, v}] of Object.entries(defaults)) {
        const col = cols[p];
        cols[p] = column(type, entries.length, (k) => entries[k][2] ? v[k] : at(col, k));
      }
    }
    const idx = entries.map(([i]) => i);
    s[name] = {t: pick(src.t, idx), valid: ones(idx.length), cols};
  }
  delete s.sensorEventsDEPRECATED;
}

function carParams(s) {
  const {cols} = s.carParams, n = ns(s.carParams);
  const mass = cols["/deprecated/mass"];
  if (mass) {
    for (const p of ["/rotationalInertiaPerMass", "/tireStiffnessFrontPerMass", "/tireStiffnessRearPerMass"]) {
      const c = cols[p];
      if (c) cols[p] = column("f32", n, (i) => at(mass, i) > 0 ? at(c, i) / at(mass, i) : at(c, i));
    }
    cols["/deprecated/mass"] = column("f32", n, (i) => at(mass, i) > 0 ? 0 : at(mass, i));
  }
  const fp = cols["/carFingerprint"];
  if (fp) cols["/carFingerprint"] = column("t", n, (i) => tables.fingerprints[at(fp, i)] ?? at(fp, i));
  for (let j = 0; cols[`/carFw/${j}/ecu`]; j++) {
    const fw = present(cols, `/carFw/${j}`, n);
    cols[`/carFw/${j}/brand`] = column("t", n, (i) => fw[i] ? at(cols["/brand"], i) ?? "" : null);
  }
}

function gpsLocation(s) {
  for (const svc of [s.gpsLocation, s.gpsLocationExternal]) {
    if (!svc) continue;
    const fix = svc.cols["/hasFix"], flags = svc.cols["/flags"];
    svc.cols["/hasFix"] = column("b", ns(svc), (i) => at(fix, i) || at(flags, i) === 1 ? 1 : 0);
  }
}

function deviceState(s, ctx) {
  if (!s.deviceState || ctx.deviceType === undefined) return;
  s.deviceState.cols["/deviceType"] = value(ctx.deviceType, "e", ns(s.deviceState), names(STRUCTS.DeviceState, "deviceType"));
}

function carOutput(s) {
  const cc = s.carControl, n = ns(cc), src = "/deprecated/actuatorsOutput";
  const st = STRUCTS["CarControl.Actuators"], exists = present(cc.cols, src, n);
  // a null actuatorsOutput copies as a struct of defaults
  const set = Object.fromEntries(st.fields.map(([name]) => [`/${name}`, (i) => exists[i] ? at(cc.cols[`${src}/${name}`], i) : 0]));
  s.carOutput = {t: cc.t, valid: cc.valid, cols: struct(st, "/actuatorsOutput", n, set)};
}

function controlsState(s) {
  const cs = s.controlsState, st = STRUCTS.SelfdriveState;
  const set = {};
  for (const f of ["enabled", "active", "state", "engageable", "alertText1", "alertText2", "alertStatus", "alertSize", "alertType", "personality"]) {
    set[`/${f}`] = cs.cols[`/deprecated/${f}`];
  }
  s.selfdriveState = {t: cs.t, valid: cs.valid, cols: struct(st, "", ns(cs), set)};
}

function carState(s) {
  const cs = s.controlsState, car = s.carState;
  if (!cs || !car) return;
  const n = ns(car), vc = cs.cols["/deprecated/vCruise"], vcc = cs.cols["/deprecated/vCruiseCluster"];
  const v = car.cols["/vCruise"], vClu = car.cols["/vCruiseCluster"];
  // the last controlsState before each carState in log order (seq), or by time for readers without it
  const order = (svc) => svc.seq ?? svc.t;
  const a = order(cs), b = order(car), last = new Int32Array(n);
  for (let i = 0, k = -1; i < n; i++) {
    while (k + 1 < a.length && a[k + 1] <= b[i]) k++;
    last[i] = k >= 0 && at(vc, k) - at(v, i) > 0.1 ? k : -1;
  }
  car.cols["/vCruise"] = column("f32", n, (i) => last[i] >= 0 ? at(vc, last[i]) : at(v, i));
  car.cols["/vCruiseCluster"] = column("f32", n, (i) => last[i] >= 0 ? at(vcc, last[i]) : at(vClu, i));
}

const nanoseconds = (svc) => (i) => Math.round(svc.t[i] * 1e9);

function liveLocationKalman(s) {
  const llk = s.liveLocationKalmanDEPRECATED, n = ns(llk), src = llk.cols;
  const set = {"/timestamp": nanoseconds(llk)};
  for (const f of ["inputsOK", "posenetOK", "sensorsOK"]) set[`/${f}`] = src[`/${f}`];
  const cols = struct(STRUCTS.DeviceMotion, "", n, set);
  for (const f of ["orientationNED", "velocityDevice", "accelerationDevice", "angularVelocityDevice"]) {
    // an empty value or std list becomes NaN, which is a missing column
    const xyz = {"/valid": src[`/${f}/valid`]};
    ["x", "y", "z"].forEach((a, k) => {
      xyz[`/${a}`] = src[`/${f}/value/${k}`] ?? null;
      xyz[`/${a}Std`] = src[`/${f}/std/${k}`] ?? null;
    });
    struct(STRUCTS["DeviceMotion.XYZMeasurement"], `/${f}`, n, xyz, null, "", cols);
  }
  s.deviceMotion = {t: llk.t, valid: llk.valid, cols};
  delete s.liveLocationKalmanDEPRECATED;
}

function deviceMotion(s) {
  const dm = s.deviceMotion, ts = dm.cols["/timestamp"];
  if (dm.t.every((_, i) => at(ts, i) === 0)) dm.cols["/timestamp"] = column("u64", ns(dm), nanoseconds(dm));
}

function liveTracks(s) {
  const lt = s.liveTracksDEPRECATED, n = ns(lt), point = STRUCTS["RadarData.RadarPoint"];
  const cols = struct(STRUCTS.RadarData, "", n);
  for (let j = 0; lt.cols[`/${j}/trackId`]; j++) {
    const set = Object.fromEntries(["trackId", "dRel", "yRel", "vRel"].map((f) => [`/${f}`, lt.cols[`/${j}/${f}`]]));
    struct(point, `/points/${j}`, n, set, present(lt.cols, `/${j}`, n), "", cols);
  }
  s.radarTracks = {t: lt.t, valid: lt.valid, cols};
  delete s.liveTracksDEPRECATED;
}

function driverAssistance(s) {
  const lp = s.longitudinalPlan;
  s.driverAssistance = {t: lp.t, valid: ones(ns(lp)), cols: struct(STRUCTS.DriverAssistance, "", ns(lp))};
}

function drivingModelData(s, ctx, missing) {
  const m = s.modelV2, n = ns(m), src = m.cols, st = STRUCTS.DrivingModelData;
  const set = {};
  for (const f of ["frameId", "frameIdExtra", "frameDropPerc", "modelExecutionTime", "big"]) set[`/${f}`] = src[`/${f}`];
  const cols = struct(st, "", n, set);
  const action = STRUCTS["ModelDataV2.Action"];
  struct(action, "/action", n, Object.fromEntries(action.fields.map(([f]) => [`/${f}`, src[`/action/${f}`]])), null, "", cols);
  struct(STRUCTS["DrivingModelData.MetaData"], "/meta", n, {"/laneChangeState": src["/meta/laneChangeState"]}, null, "", cols);

  // fill_lane_line_meta reads laneLines[1].y[0] and laneLines[2].y[0]; laneLines[*].y are 33 long
  if (src["/laneLines/1/y/0"] && src["/laneLines/2/y/0"]) {
    const ok = Uint8Array.from({length: n}, (_, i) => has(src["/laneLines/0/y/0"], i) && has(src["/laneLineProbs/0"], i));
    const val = (p) => (i) => ok[i] ? at(src[p], i) : NaN;
    Object.assign(cols, struct(STRUCTS["DrivingModelData.LaneLineMeta"], "/laneLineMeta", n, {
      "/leftY": val("/laneLines/1/y/0"), "/leftProb": val("/laneLineProbs/1"), "/rightY": val("/laneLines/2/y/0"), "/rightProb": val("/laneLineProbs/2"),
    }));
  } else if (dropped(m, "/laneLines/1/y")) {
    missing.push("drivingModelData/laneLineMeta");
  }

  // fill_xyz_poly: a least-squares fit of position over T_IDXS, which is a fixed linear map of the 33 samples
  if (["x", "y", "z"].every((a) => src[`/position/${a}/32`])) {
    const ok = Uint8Array.from({length: n}, (_, i) => has(src["/position/x/0"], i) && has(src["/position/y/0"], i) && has(src["/position/z/0"], i));
    for (const a of ["x", "y", "z"]) {
      const xs = T_IDXS.map((_, k) => src[`/position/${a}/${k}`]);
      tables.polyfit.forEach((row, d) => {
        cols[`/path/${a}Coefficients/${d}`] = column("f32", n, (i) => ok[i] ? row.reduce((acc, w, k) => acc + w * at(xs[k], i), 0) : NaN);
      });
    }
  } else if (dropped(m, "/position/x")) {
    missing.push("drivingModelData/path");
  }
  s.drivingModelData = {t: m.t, valid: m.valid, cols};
}

function onroadEvents(s) {
  const old = s.onroadEventsDEPRECATED, n = ns(old);
  const st = STRUCTS.OnroadEvent, newNames = names(st, "name");
  const to = names(STRUCTS.OnroadEventDEPRECATED, "name").map((x) => x.endsWith("DEPRECATED") ? -1 : newNames.indexOf(x));
  // per event, the source items that survive: names that still exist and aren't deprecated
  const kept = Array.from({length: n}, (_, i) => {
    const r = [];
    for (let j = 0; has(old.cols[`/${j}/name`], i); j++) if (to[at(old.cols[`/${j}/name`], i)] >= 0) r.push(j);
    return r;
  });
  const cols = {};
  for (let k = 0; k < Math.max(0, ...kept.map((r) => r.length)); k++) {
    const src = (f, map = (x) => x) => (i) => k < kept[i].length ? map(at(old.cols[`/${kept[i][k]}/${f}`], i)) : NaN;
    const set = Object.fromEntries(st.fields.map(([f]) => [`/${f}`, src(f)]));
    set["/name"] = src("name", (x) => to[x]);
    struct(st, `/${k}`, n, set, null, "", cols);
  }
  s.onroadEvents = {t: old.t, valid: old.valid, cols};
  delete s.onroadEventsDEPRECATED;
}

function driverMonitoringState(s) {
  const old = s.driverMonitoringStateDEPRECATED, n = ns(old), st = STRUCTS.DriverMonitoringState;
  const get = (f) => (i) => at(old.cols[`/${f}`], i);
  const [active, status, awareActive, awarePassive, step] = ["isActiveMode", "awarenessStatus", "awarenessActive", "awarenessPassive", "stepChange"].map(get);
  const [pitchCount, yawCount, hiStd, uncertain, distracted] = ["posePitchValidCount", "poseYawValidCount", "hiStdCount", "uncertainCount", "distractedType"].map(get);
  const percent = (x) => Math.trunc(Math.max(0, Math.min(100, x)));
  const eventNames = names(STRUCTS.OnroadEvent, "name");
  const events = (i) => {
    const r = [];
    for (let j = 0; has(old.cols[`/events/${j}/name`], i); j++) r.push(eventNames[at(old.cols[`/events/${j}/name`], i)]);
    return r;
  };
  const levels = names(st, "alertLevel"), level = {driverDistracted1: "one", driverUnresponsive1: "one", driverDistracted2: "two",
    driverUnresponsive2: "two", driverDistracted3: "three", driverUnresponsive3: "three"};
  const policies = names(st, "activePolicy");

  const cols = struct(st, "", n, {
    "/isRHD": old.cols["/isRHD"],
    "/activePolicy": (i) => policies.indexOf(active(i) ? "vision" : "wheeltouch"),
    "/alertLevel": (i) => levels.indexOf(level[events(i).find((e) => level[e])] ?? "none"),
    "/lockout": (i) => events(i).includes("tooDistracted") ? 1 : 0,
  });
  const vision = STRUCTS["DriverMonitoringState.VisionPolicyState"], pose = STRUCTS["DriverMonitoringState.VisionPolicyState.Pose"];
  const calib = STRUCTS["DriverMonitoringState.CalibrationState"];
  struct(vision, "/visionPolicyState", n, {
    "/awarenessPercent": (i) => percent((active(i) ? status(i) : awareActive(i)) * 100),
    "/awarenessStep": (i) => active(i) ? step(i) : 0,
    "/isDistracted": old.cols["/isDistracted"],
    "/faceDetected": old.cols["/faceDetected"],
    "/wheeltouchFallbackPercent": (i) => percent(hiStd(i) / 200 * 100),
    "/uncertainOffroadAlertPercent": (i) => percent(uncertain(i) / 1200 * 100),
  }, null, "", cols);
  struct(STRUCTS["DriverMonitoringState.VisionPolicyState.DistractedTypes"], "/visionPolicyState/distractedTypes", n, {
    "/pose": (i) => distracted(i) & 1 ? 1 : 0, "/eye": (i) => distracted(i) & 2 ? 1 : 0, "/phone": (i) => distracted(i) & 4 ? 1 : 0,
  }, null, "", cols);
  struct(pose, "/visionPolicyState/pose", n, {"/calibrated": (i) => pitchCount(i) >= 1200 && yawCount(i) >= 1200 ? 1 : 0}, null, "", cols);
  struct(calib, "/visionPolicyState/pose/pitchCalib", n, {
    "/offset": old.cols["/posePitchOffset"], "/calibratedPercent": (i) => percent(pitchCount(i) / 1200 * 100),
  }, null, "", cols);
  struct(calib, "/visionPolicyState/pose/yawCalib", n, {
    "/offset": old.cols["/poseYawOffset"], "/calibratedPercent": (i) => percent(yawCount(i) / 1200 * 100),
  }, null, "", cols);
  struct(STRUCTS["DriverMonitoringState.WheeltouchPolicyState"], "/wheeltouchPolicyState", n, {
    "/awarenessPercent": (i) => percent((active(i) ? awarePassive(i) : status(i)) * 100),
    "/awarenessStep": (i) => active(i) ? 0 : step(i),
  }, null, "", cols);
  s.driverMonitoringState = {t: old.t, valid: old.valid, cols};
  delete s.driverMonitoringStateDEPRECATED;
}

const CONTROL_N = 17, DT_MDL = 0.05, MIN_STABLE_DELAY = 0.3;

function longitudinalPlan(s, ctx, missing) {
  const lp = s.longitudinalPlan, n = lp && ns(lp);
  if (!lp || ctx.carFingerprint === undefined || !lp.t.every((_, i) => at(lp.cols["/aTarget"], i) === 0)) return;
  const speed = (j) => lp.cols[`/speeds/${j}`], accel0 = lp.cols["/accels/0"];
  if (dropped(lp, "/speeds") || dropped(lp, "/accels")) return missing.push("longitudinalPlan/aTarget");
  // get_accel_from_plan with action_t = DT_MDL, which only needs speeds 0, 5 and 6
  const full = (i) => has(speed(CONTROL_N - 1), i) && !has(speed(CONTROL_N), i);
  const aTarget = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    if (!full(i)) continue;
    const v = (j) => at(speed(j), i), vNow = v(0);
    const vStable = (v(6) - v(5)) / (T_IDXS[6] - T_IDXS[5]) * (MIN_STABLE_DELAY - T_IDXS[5]) + v(5);  // np.interp
    const vTarget = vNow + (DT_MDL / MIN_STABLE_DELAY) * (vStable - vNow);
    aTarget[i] = 2 * (vTarget - vNow) / DT_MDL - at(accel0, i);
  }
  lp.cols["/aTarget"] = column("f32", n, (i) => aTarget[i]);
  lp.cols["/shouldStop"] = column("b", n, (i) => (full(i) ? at(speed(0), i) : 0) < 0.3 && aTarget[i] < 0.1 ? 1 : 0);
}

// [inputs, product, fn] in migrate_all order. As in python, each sees the segment's original services and is skipped
// when its product already exists. migrate_all's opt-in ones (manager_states=...) are left out, as by its defaults.
const MIGRATIONS = [
  [["sensorEventsDEPRECATED"], "sensorEvents", sensorEvents],
  [["carParams"], null, carParams],
  [["gpsLocation", "gpsLocationExternal"], null, gpsLocation],
  [["deviceState"], null, deviceState],
  [["carControl"], "carOutput", carOutput],
  [["controlsState"], "selfdriveState", controlsState],
  [["carState"], null, carState],
  [["liveLocationKalmanDEPRECATED"], "deviceMotion", liveLocationKalman],
  [["deviceMotion"], null, deviceMotion],
  [["liveTracksDEPRECATED"], "radarTracks", liveTracks],
  [["longitudinalPlan"], "driverAssistance", driverAssistance],
  [["modelV2"], "drivingModelData", drivingModelData],
  [["onroadEventsDEPRECATED"], "onroadEvents", onroadEvents],
  [["driverMonitoringStateDEPRECATED"], null, driverMonitoringState],
  [["longitudinalPlan"], null, longitudinalPlan],
];

// fields the schema moved with no python migration, fixed in logs from before the move. An old log is told by its
// data in the old field, which new code never writes.
const FIXES = [
  // 069506aa36 (2026-06): selfdriveState.alertSound moved from @8 (now deprecated.alertSound) to @13; same ordinals
  (s) => {
    const cols = s.selfdriveState?.cols, old = cols?.["/deprecated/alertSound"];
    if (old?.v.some((x) => x > 0)) cols["/alertSound"] = {...cols["/alertSound"], v: old.v};
  },
];

// the reader's keep option plus the lists the migrations read
export function withKeep(keep = {}) {
  const out = {...keep};
  for (const [name, globs] of Object.entries(KEEP)) out[name] = [...(out[name] || []), ...globs];
  return out;
}

// once the migrations have read them, the long KEEP lists that weren't asked for are dropped again and reported as
// skipped (svc.lists), as if read without them
function unkeep(s, keep = {}) {
  for (const [name, globs] of Object.entries(KEEP)) {
    const svc = s[name];
    if (!svc) continue;
    const ours = pattern(globs), theirs = keep[name] && pattern(keep[name]);
    const lists = new Map();  // list path → [length, its columns]
    for (const p of Object.keys(svc.cols)) {
      const parts = p.split("/");
      const k = parts.findIndex((x, k) => k > 0 && /^\d+$/.test(x) && ours.test(parts.slice(0, k).join("/")));
      if (k < 0) continue;
      const list = parts.slice(0, k).join("/"), e = lists.get(list) || lists.set(list, [0, []]).get(list);
      e[0] = Math.max(e[0], +parts[k] + 1);
      e[1].push(p);
    }
    for (const [list, [length, paths]] of lists) {
      if (length <= MAX_LIST || theirs?.test(list)) continue;
      for (const p of paths) delete svc.cols[p];
      svc.lists?.set(list, length);
    }
  }
}

// migrates the reader's Map(which → {name, t, valid, cols: Map, lists}) in place; read it with {keep: withKeep(keep)}
// and pass the same keep in options. Returns this segment's context, for routeContext of later segments, and the
// derived paths that could not be computed.
export function migrate(services, routeContext = {}, options = {}) {
  const s = {};
  for (const svc of services.values()) s[svc.name] = {...svc, cols: Object.fromEntries(svc.cols)};
  const own = context(s), ctx = {...routeContext, ...own}, missing = [], original = new Set(Object.keys(s));
  for (const [inputs, product, fn] of MIGRATIONS) {
    if (!original.has(product) && inputs.some((name) => original.has(name))) fn(s, ctx, missing);
  }
  for (const fix of FIXES) fix(s);
  unkeep(s, options.keep);
  services.clear();
  for (const [name, svc] of Object.entries(s)) services.set(name, {lists: new Map(), ...svc, name, cols: new Map(Object.entries(svc.cols))});
  return {context: own, missing};
}
