// a video track's stream stats as connect's stream view shows them (BodyTeleop/StatusBar.jsx): frame rate, bitrate,
// round trip time and packet loss, from RTCPeerConnection.getStats() reports about a second apart, and the connection
// called poor when it loses packets, lags or the video stops
const LOSS_POOR = 0.02, RTT_POOR_MS = 250;

// report: an RTCStatsReport (a Map); last: this function's result for the report before, or null; track: the
// MediaStreamTrack id, else the first video. → {fps, bitrate (bit/s), rtt (ms), loss (0..1), poor, ...the counters}
export function sample(report, last, track) {
  let video = null, pair = null;
  for (const s of report.values()) {
    if (s.type === "inbound-rtp" && s.kind === "video" && (!video || s.trackIdentifier === track)) video = s;
    if (s.type === "transport" && s.selectedCandidatePairId) pair = report.get(s.selectedCandidatePairId);
  }
  // browsers that don't name the transport's pair
  if (!pair) for (const s of report.values()) if (s.type === "candidate-pair" && s.nominated && s.state === "succeeded") pair = s;
  const rtt = pair?.currentRoundTripTime != null ? pair.currentRoundTripTime * 1000 : null;
  const started = !!last?.started;
  // the video stopped coming (a track that went away), or never did
  if (!video) return {rtt, fps: null, bitrate: null, loss: null, started, poor: started || rtt > RTT_POOR_MS};

  const out = {rtt, fps: 0, bitrate: 0, loss: 0, started, t: video.timestamp, bytes: video.bytesReceived, frames: video.framesDecoded,
    lost: video.packetsLost ?? last?.lost, received: video.packetsReceived ?? last?.received};
  const dt = (out.t - last?.t) / 1000;
  if (last?.t != null && dt > 0) {
    out.bitrate = (out.bytes - last.bytes) * 8 / dt;
    out.fps = (out.frames - last.frames) / dt;
    const lost = out.lost - last.lost, total = lost + out.received - last.received;
    if (total > 0) out.loss = lost / total;
    out.started ||= out.bitrate > 0;
  }
  out.poor = (out.started && out.bitrate === 0) || out.loss > LOSS_POOR || rtt > RTT_POOR_MS;
  return out;
}

// as connect writes them; "--" for what isn't known
export const format = (s) => ({
  fps: s?.fps != null ? s.fps.toFixed(1) : "--",
  bitrate: s?.bitrate == null ? "--" : s.bitrate > 1e6 ? `${(s.bitrate / 1e6).toFixed(2)} Mbps` : `${(s.bitrate / 1e3).toFixed(0)} kbps`,
  rtt: s?.rtt != null ? `${Math.round(s.rtt)} ms` : "--",
  loss: s?.loss != null ? `${(s.loss * 100).toFixed(1)}%` : "--",
});
