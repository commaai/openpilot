// node --test data/live/rtcstats.test.mjs: stream stats from fake getStats() reports
import assert from "node:assert/strict";
import {test} from "node:test";
import {format, sample} from "./rtcstats.js";

const report = (...stats) => new Map(stats.map((s, i) => [s.id || `s${i}`, s]));
const video = (t, bytes, frames, lost, received, track = "a") =>
  ({type: "inbound-rtp", kind: "video", trackIdentifier: track, timestamp: t, bytesReceived: bytes, framesDecoded: frames, packetsLost: lost, packetsReceived: received});
const pair = (rtt) => [{type: "transport", selectedCandidatePairId: "p"}, {id: "p", type: "candidate-pair", currentRoundTripTime: rtt}];

test("rates over the time between reports, the transport's pair's round trip", () => {
  const a = sample(report(video(1000, 0, 0, 0, 0), ...pair(0.05)), null, "a");
  assert.deepEqual([a.fps, a.bitrate, a.rtt, a.poor], [0, 0, 50, false]);
  const b = sample(report(video(3000, 500e3, 40, 1, 199), ...pair(0.0234)), a, "a");
  assert.equal(b.fps, 20);
  assert.equal(b.bitrate, 2e6);
  assert.equal(b.loss, 1 / 200);
  assert.deepEqual(format(b), {fps: "20.0", bitrate: "2.00 Mbps", rtt: "23 ms", loss: "0.5%"});
  assert.equal(format({bitrate: 400e3}).bitrate, "400 kbps");
  assert.deepEqual(format(null), {fps: "--", bitrate: "--", rtt: "--", loss: "--"});
});

test("poor: loss over 2%, a round trip over 250 ms, or the video stopping once it came", () => {
  const a = sample(report(video(0, 0, 0, 0, 0)), null);
  assert.equal(sample(report(video(1000, 1e5, 20, 0, 100)), a).poor, false);
  assert.equal(sample(report(video(1000, 1e5, 20, 3, 97)), a).poor, true);
  assert.equal(sample(report(video(1000, 1e5, 20, 0, 100), ...pair(0.3)), a).poor, true);
  const on = sample(report(video(1000, 1e5, 20, 0, 100)), a), stalled = sample(report(video(2000, 1e5, 20, 0, 100)), on);
  assert.equal(stalled.poor, true);
  assert.equal(sample(report(), on).poor, true);
  assert.equal(sample(report(), null).poor, false);  // nothing yet
});

test("the asked track's stats among several, a nominated pair without a transport's", () => {
  const s = sample(report(video(0, 0, 0, 0, 0, "a"), video(0, 9, 9, 0, 0, "b"),
    {type: "candidate-pair", nominated: true, state: "succeeded", currentRoundTripTime: 0.01}), null, "b");
  assert.deepEqual([s.bytes, s.rtt], [9, 10]);
});
