import { handleKeyX, executePlan } from "./controls.js";
import { start, stop, lastChannelMessageTime, playSoundRequest, getGeminiStatus, setGeminiStatus } from "./webrtc.js";

export var pc = null;
export var dc = null;
export var geminiEnabled = false;

// Keyboard input handler - disabled when Gemini is enabled
function handleKeyboardInput(e, value) {
  if (!geminiEnabled) {
    handleKeyX(e.key.toLowerCase(), value);
  }
}

document.addEventListener('keydown', (e)=>(handleKeyboardInput(e, 1)));
document.addEventListener('keyup', (e)=>(handleKeyboardInput(e, 0)));
$(".keys").bind("mousedown touchstart", (e)=>{
  if (!geminiEnabled) {
    handleKeyX($(e.target).attr('id').replace('key-', ''), 1);
  }
});
$(".keys").bind("mouseup touchend", (e)=>{
  if (!geminiEnabled) {
    handleKeyX($(e.target).attr('id').replace('key-', ''), 0);
  }
});
$("#plan-button").click(executePlan);
$(".sound").click((e)=>{
  const sound = $(e.target).attr('id').replace('sound-', '')
  return playSoundRequest(sound);
});

// Gemini control toggle
$("#gemini-toggle").change(function() {
  const toggle = $(this);
  const enabled = toggle.is(':checked');
  geminiEnabled = enabled;
  setGeminiStatus(enabled).then(function(data) {
    $("#gemini-status").text(enabled ? "Enabled" : "Disabled");
    if (enabled) {
      // Clear keyboard input when Gemini takes over
      handleKeyX('w', 0);
      handleKeyX('a', 0);
      handleKeyX('s', 0);
      handleKeyX('d', 0);
      $("#key-w, #key-a, #key-s, #key-d").css('background', '#333');
      $("#pos-vals").text("0,0");
    }
  }).catch(function(err) {
    console.error("Error toggling Gemini control:", err);
    // Revert toggle on error
    toggle.prop('checked', !enabled);
    geminiEnabled = false;
  });
});

// Load initial Gemini status
getGeminiStatus().then(function(data) {
  const enabled = data.enabled || false;
  geminiEnabled = enabled;
  $("#gemini-toggle").prop('checked', enabled);
  $("#gemini-status").text(enabled ? "Enabled" : "Disabled");
}).catch(function(err) {
  console.error("Error getting Gemini status:", err);
});

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
