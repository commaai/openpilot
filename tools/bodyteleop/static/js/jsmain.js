import { handleKeyX, executePlan } from "./controls.js";
import { start, stop, last_ping } from "./webrtc.js";

export var pc = null;
export var dc = null;

document.addEventListener('keydown', (e)=>(handleKeyX(e.key.toLowerCase(), 1)));
document.addEventListener('keyup', (e)=>(handleKeyX(e.key.toLowerCase(), 0)));
$(".keys").bind("mousedown touchstart", (e)=>handleKeyX($(e.target).attr('id').replace('key-', ''), 1));
$(".keys").bind("mouseup touchend", (e)=>handleKeyX($(e.target).attr('id').replace('key-', ''), 0));
$("#plan-button").click(executePlan);

setInterval( () => {
  const dt = new Date().getTime();
  if ((dt - last_ping) > 1000) {
    $(".pre-blob").removeClass('blob');
    $("#battery").text("-");
    $("#ping-time").text('-');
    $("video")[0].load();
  }
}, 5000);

start(pc, dc);