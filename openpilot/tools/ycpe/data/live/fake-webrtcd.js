// a webrtcd stand-in in the page, for tests and ?fake-webrtcd on /live/local: a peer connection of its own answers the
// offer and sends one video track, a test pattern of the camera it shows, which livestreamCameraSwitch switches as a
// stock webrtcd's one track does. It keeps every message the page sends on "data" (window.__webrtcd.received), so a
// test can check what a device would have been told.
const COLORS = {road: "#1565c0", wideRoad: "#2e7d32", driver: "#e65100"};

export async function stream({sdp, cameras}) {
  const fake = window.__webrtcd = {received: [], camera: cameras[0] || "road"};
  const canvas = Object.assign(document.createElement("canvas"), {width: 640, height: 360});
  const g = canvas.getContext("2d");
  let frame = 0;
  const draw = setInterval(() => {
    g.fillStyle = COLORS[fake.camera] || "#555";
    g.fillRect(0, 0, 640, 360);
    g.fillStyle = "#fff";
    g.font = "bold 48px sans-serif";
    g.fillText(fake.camera, 40, 120);
    g.fillText(String(frame++), 40, 200);
  }, 50);
  const pc = new RTCPeerConnection();
  pc.onconnectionstatechange = () => ["failed", "closed"].includes(pc.connectionState) && clearInterval(draw);
  pc.ondatachannel = ({channel}) => {
    if (channel.label !== "data") return;
    channel.onmessage = ({data}) => {
      const msg = JSON.parse(data);
      fake.received.push(msg);
      if (msg.type === "livestreamCameraSwitch") fake.camera = msg.data.camera;
    };
  };
  await pc.setRemoteDescription({type: "offer", sdp});
  const [video] = pc.getTransceivers().filter((t) => t.receiver.track.kind === "video");
  if (video) {
    video.direction = "sendonly";
    await video.sender.replaceTrack(canvas.captureStream(20).getVideoTracks()[0]);
  }
  await pc.setLocalDescription(await pc.createAnswer());
  // no trickled candidates, as webrtcd
  await new Promise((done) => {
    pc.addEventListener("icegatheringstatechange", () => pc.iceGatheringState === "complete" && done());
    if (pc.iceGatheringState === "complete") done();
  });
  fake.pc = pc;
  return {sdp: pc.localDescription.sdp, type: "answer"};
}
