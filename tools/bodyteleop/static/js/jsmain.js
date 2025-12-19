import { handleKeyX, executePlan, setGeminiXY, setGeminiEnabled, getXY } from "./controls.js";
import { start, stop, lastChannelMessageTime, playSoundRequest, getGeminiStatus, setGeminiStatus, getGeminiPrompt, setGeminiPrompt } from "./webrtc.js";

export var pc = null;
export var dc = null;
export var geminiEnabled = false;

// Execute the current Gemini plan locally in the browser.
// This is required because joystick commands are sent via WebRTC data channel (getXY() every 50ms).
let lastGeminiPlanId = null;
let geminiPlanTimer = null;
let currentGeminiPlan = null;
let currentGeminiPlanStartMs = null;

function stopGeminiPlanExecution() {
  if (geminiPlanTimer !== null) {
    clearInterval(geminiPlanTimer);
    geminiPlanTimer = null;
  }
  currentGeminiPlan = null;
  currentGeminiPlanStartMs = null;
  setGeminiXY(0, 0);
}

function startGeminiPlanExecution(plan) {
  // plan: [[w,a,s,d,t], ...] where t is cumulative seconds from plan start
  stopGeminiPlanExecution();
  currentGeminiPlan = plan;
  currentGeminiPlanStartMs = performance.now();

  geminiPlanTimer = setInterval(() => {
    if (!geminiEnabled || !currentGeminiPlan || currentGeminiPlan.length === 0) {
      stopGeminiPlanExecution();
      return;
    }

    const elapsedS = (performance.now() - currentGeminiPlanStartMs) / 1000.0;

    // Find current step
    let step = null;
    for (let i = 0; i < currentGeminiPlan.length; i++) {
      const s = currentGeminiPlan[i];
      const t = s[4];
      if (elapsedS <= t) {
        step = s;
        break;
      }
    }

    if (step === null) {
      // Finished
      stopGeminiPlanExecution();
      return;
    }

    const [w, a, s, d, _t] = step;
    const x = (w || 0) - (s || 0);
    const y = (d || 0) - (a || 0);
    setGeminiXY(x, y);
    $("#pos-vals").text(`${x},${y}`);
  }, 50);
}

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
  setGeminiEnabled(enabled); // Update controls.js
  $("#gemini-status-text").text(enabled ? "⏳ Enabling..." : "⏳ Disabling...");
  setGeminiStatus(enabled).then(function(data) {
    $("#gemini-status").text(enabled ? "Enabled" : "Disabled");
    updateGeminiStatus(); // Refresh status display
    if (enabled) {
      // Clear keyboard input when Gemini takes over
      handleKeyX('w', 0);
      handleKeyX('a', 0);
      handleKeyX('s', 0);
      handleKeyX('d', 0);
      $("#key-w, #key-a, #key-s, #key-d").css('background', '#333');
      $("#pos-vals").text("0,0");
    } else {
      stopGeminiPlanExecution();
    }
  }).catch(function(err) {
    console.error("Error toggling Gemini control:", err);
    // Revert toggle on error
    toggle.prop('checked', !enabled);
    geminiEnabled = false;
    setGeminiEnabled(false);
    stopGeminiPlanExecution();
    $("#gemini-status-text").text("✗ Error");
  });
});

// Function to update Gemini status display
function updateGeminiStatus() {
  getGeminiStatus().then(function(data) {
    const enabled = data.enabled || false;
    geminiEnabled = enabled;
    setGeminiEnabled(enabled); // Update controls.js
    $("#gemini-toggle").prop('checked', enabled);
    $("#gemini-status").text(enabled ? "Enabled" : "Disabled");

    // Update status display
    $("#gemini-status-text").text(enabled ? "✓ Active" : "✗ Inactive");
    $("#gemini-api-key-status").text(data.api_key_set ? "✓ Set" : "✗ Not Set");
    $("#gemini-command-status").text(`x=${(data.current_x || 0).toFixed(2)}, y=${(data.current_y || 0).toFixed(2)}`);
    $("#gemini-response-status").text(data.last_response || "-");

    // Update Gemini XY values for controls.js
    if (enabled) {
      // Start executing the plan client-side when a new plan arrives.
      // This drives getXY() at WebRTC send rate (50ms) without requiring server polling.
      if (data.current_plan && data.current_plan.length > 0) {
        const planId = data.current_plan_id;
        if (lastGeminiPlanId === null || planId !== lastGeminiPlanId) {
          lastGeminiPlanId = planId;
          startGeminiPlanExecution(data.current_plan);
        }
      } else {
        // No plan: ensure we aren't running an old one
        stopGeminiPlanExecution();
      }
    } else {
      // When disabled, update pos-vals from keyboard input
      stopGeminiPlanExecution();
      const {x, y} = getXY();
      $("#pos-vals").text(`${x},${y}`);
    }

    // Display current plan
    if (data.current_plan && data.current_plan.length > 0) {
      const planStr = data.current_plan.map((step, idx) => {
        const [w, a, s, d, t] = step;
        return `${idx + 1}. W=${w} A=${a} S=${s} D=${d} until ${t.toFixed(2)}s`;
      }).join('\n');
      $("#gemini-plan-status").text(planStr);
    } else {
      $("#gemini-plan-status").text("-");
    }

    // Load prompt if available
    if (data.prompt) {
      $("#gemini-prompt").val(data.prompt);
    }
  }).catch(function(err) {
    console.error("Error getting Gemini status:", err);
    $("#gemini-status-text").text("✗ Error");
  });
}

// Load initial Gemini status and prompt
updateGeminiStatus();

// Update status every 2 seconds
setInterval(updateGeminiStatus, 2000);

// Load prompt separately in case status doesn't include it
getGeminiPrompt().then(function(data) {
  if (data.prompt) {
    $("#gemini-prompt").val(data.prompt);
  }
}).catch(function(err) {
  console.error("Error getting Gemini prompt:", err);
});

// Save prompt button
$("#gemini-prompt-save").click(function() {
  const prompt = $("#gemini-prompt").val();
  setGeminiPrompt(prompt).then(function(data) {
    $("#gemini-prompt-status").text("Saved!").css("color", "green");
    setTimeout(function() {
      $("#gemini-prompt-status").text("");
    }, 2000);
  }).catch(function(err) {
    console.error("Error saving Gemini prompt:", err);
    $("#gemini-prompt-status").text("Error saving").css("color", "red");
  });
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
