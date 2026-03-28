import { handleKeyX, executePlan } from "./controls.js";
import { start, stop, lastChannelMessageTime, playSoundRequest, setAutonomyEnabled, setAutonomyConfig, getAutonomyStatus } from "./webrtc.js";

export var pc = null;
export var dc = null;

document.addEventListener('keydown', (e)=>(handleKeyX(e.key.toLowerCase(), 1)));
document.addEventListener('keyup', (e)=>(handleKeyX(e.key.toLowerCase(), 0)));
$(".keys").bind("mousedown touchstart", (e)=>handleKeyX($(e.target).attr('id').replace('key-', ''), 1));
$(".keys").bind("mouseup touchend", (e)=>handleKeyX($(e.target).attr('id').replace('key-', ''), 0));
$("#plan-button").click(executePlan);
$(".sound").click((e)=>{
  const sound = $(e.target).attr('id').replace('sound-', '')
  return playSoundRequest(sound);
});

$("#autonomy-enable").click(() => setAutonomyEnabled(true));
$("#autonomy-disable").click(() => setAutonomyEnabled(false));
$("#autonomy-config").click(() => {
  return setAutonomyConfig({
    target_visible: $("#target-visible").is(":checked"),
    target_distance_m: Number($("#target-distance").val()),
    target_bearing_deg: Number($("#target-bearing").val()),
    obstacle_distance_m: Number($("#obstacle-distance").val()),
  });
});

setInterval(() => {
  getAutonomyStatus()
    .then((r) => r.json())
    .then((resp) => {
      const status = resp.status || {};
      const axes = status.axes || [0, 0];
      $("#autonomy-status").text(`state: ${status.state || '-'} | attending: ${status.attending} | visible: ${status.target_visible} | axes: ${axes[0].toFixed(2)}, ${axes[1].toFixed(2)}`);
    });
}, 500);

setInterval( () => {
  const dt = new Date().getTime();
  if ((dt - lastChannelMessageTime) > 1000) {
    $(".pre-blob").removeClass('blob');
    $("#battery").text("-");
    $("#ping-time").text('-');
    $("video")[0].load();
  }
}, 5000);

start(pc, dc);
