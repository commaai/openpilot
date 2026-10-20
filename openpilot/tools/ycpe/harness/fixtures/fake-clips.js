// a stand-in for a device's video clips (athenad's VideoClips) inside the page, for the clips scenarios. getClipState and
// getClipChunk are answered here from window.__clips; createClip and deleteClip go on to the network, where the
// scenario's routes answer them and __sent records them, and change the clips once answered. A new clip is queued
// in the first getClipState after it, encoding in the next and ready in the one after; every ready clip is fixtures/clip.mp4, served in
// CHUNK-byte chunks a little apart so the download's progress shows
const ATHENA = /(\/local\/athena\/|athena\.comma\.ai\/)[0-9a-f]{16}$/;
const MP4 = "https://ycpe-fixture.invalid/clip.mp4", CHUNK = 8192, DELAY = 400;

export function install({clips = [], cameras = {}} = {}) {
  const fake = window.__clips = {clips, cameras, calls: []};
  const answered = new WeakMap();  // clip → the getClipState answers it was in, so the first shows it queued
  const mp4 = fetch(MP4).then((r) => r.arrayBuffer()).then((b) => new Uint8Array(b));
  const answer = (result) => new Response(JSON.stringify({id: 0, jsonrpc: "2.0", result}), {headers: {"content-type": "application/json"}});
  const next = window.fetch;
  window.fetch = async (input, init = {}) => {
    const url = String(input.url || input), call = ATHENA.test(url) && JSON.parse(init.body || "null");
    if (!call) return next(input, init);
    const {method, params} = call;
    if (method === "getClipState") {
      fake.calls.push(call);
      for (const c of fake.clips) {
        if (c.status === "ready") continue;
        answered.set(c, (answered.get(c) ?? 0) + 1);
        if (answered.get(c) === 2) c.status = "encoding";
        if (answered.get(c) === 3) Object.assign(c, {status: "ready", size: (await mp4).length});
      }
      const route = params?.route?.split("|")[1];
      return answer({clips: structuredClone(fake.clips), ...route && {route, cameras: fake.cameras}});
    }
    if (method === "getClipChunk") {
      fake.calls.push(call);
      const data = await mp4;
      await new Promise((done) => setTimeout(done, DELAY));
      const part = data.subarray(params.offset, params.offset + CHUNK);
      return answer({data: btoa(String.fromCharCode(...part)), offset: params.offset, size: data.length});
    }
    const resp = await next(input, init);
    if (resp.ok && method === "createClip") {
      const {route, source_start_time, source_end_time, clip} = params;
      fake.clips.unshift({route: route.split("|")[1], source_start_time, source_end_time, ...clip, requested_at: Date.now() / 1000, status: "queued"});
    }
    if (resp.ok && method === "deleteClip") fake.clips = fake.clips.filter((c) => c.filename !== params.filename);
    return resp;
  };
}
