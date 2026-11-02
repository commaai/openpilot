// the account's devices: one record per device, which both modes' lists, pages and panes share, and what either mode's
// settings do to one (rename, share, unpair). onDeviceChange(fn) is told fn(dongleId, left) of each change to a device:
// its record changed in place (renamed), or it left the account (left true)
import {api, auth} from "./api.js";
import {listeners} from "./dom.js";

const changes = listeners();
const records = new Map();  // dongle id → its record
let list = null;  // the account's devices as last listed; null: to be asked for
let gen = 0;  // the account's: a list asked for under another one isn't kept

// a fresh read of a device lands in the record already shown
const record = (d) => records.has(d.dongle_id) ? Object.assign(records.get(d.dongle_id), d) : (records.set(d.dongle_id, d), d);
auth.on(() => {
  gen++;
  records.clear();
  list = null;
});

export const onDeviceChange = changes.on;

// the account's devices as last listed, or null until they are (or once they're to be asked for again)
export const deviceList = () => list;

// the list asked for afresh, the records already shown kept current
export async function listDevices() {
  const g = gen, fresh = await api.devices();
  if (g === gen) list = fresh.map(record);
  return list;
}

// the next page asks for the list again (a device was paired)
export const relistDevices = () => { list = null; };

// a device's record: the listed one (the list is asked for again every minute), else read on its own
export async function deviceRecord(dongleId) {
  return list?.find((d) => d.dongle_id === dongleId) ?? record(await api.device(dongleId));
}

// a new name ("" for none: it's named by its type)
export async function renameDevice(device, alias) {
  device.alias = (await api.setDeviceAlias(device.dongle_id, alias.trim())).alias;
  changes.emit(device.dongle_id);
}

// read access for another user, by email or user id
export const shareDevice = (device, email) => api.addDeviceUser(device.dongle_id, email.trim());

// for good, with everyone it's shared with and its prime. A refusal throws the API's reason, marked refused
export async function unpairDevice(device) {
  const resp = await api.unpairDevice(device.dongle_id);
  if (!resp?.success) throw Object.assign(new Error(resp?.error || "Could not successfully unpair"), {refused: true});
  deviceLeft(device.dongle_id);
}

// the device left the account (unpaired, or the user's access removed): its record goes, and the list is asked again
export function deviceLeft(dongleId) {
  records.delete(dongleId);
  api.device.forget(dongleId);
  list = null;
  changes.emit(dongleId, true);
}
