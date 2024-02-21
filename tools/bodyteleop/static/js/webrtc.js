import { getXY } from "./controls.js";
import { pingPoints, batteryPoints, chartPing, chartBattery } from "./plots.js";

export let controlCommandInterval = null;
export let latencyInterval = null;
export let lastChannelMessageTime = null;


export function offerRtcRequest(sdp, type) {
  return fetch('/offer', {
    body: JSON.stringify({sdp: sdp, type: type}),
    headers: {'Content-Type': 'application/json'},
    method: 'POST'
  });
}


export function playSoundRequest(sound) {
  return fetch('/sound', {
    body: JSON.stringify({sound}),
    headers: {'Content-Type': 'application/json'},
    method: 'POST'
  });
}


export function pingHeadRequest() {
  return fetch('/', {
    method: 'HEAD'
  });
}


export function createPeerConnection(pc) {
  var config = {
    sdpSemantics: 'unified-plan'
  };

  pc = new RTCPeerConnection(config);

  // connect audio / video
  pc.addEventListener('track', function(evt) {
    console.log("Adding Tracks!")
    if (evt.track.kind == 'video')
      document.getElementById('video').srcObject = evt.streams[0];
    else
      document.getElementById('audio').srcObject = evt.streams[0];
  });
  return pc;
}


export function negotiate(pc) {
  return pc.createOffer({offerToReceiveAudio:true, offerToReceiveVideo:true}).then(function(offer) {
    return pc.setLocalDescription(offer);
  }).then(function() {
    return new Promise(function(resolve) {
      if (pc.iceGatheringState === 'complete') {
        resolve();
      }
      else {
        function checkState() {
          if (pc.iceGatheringState === 'complete') {
            pc.removeEventListener('icegatheringstatechange', checkState);
            resolve();
          }
        }
        pc.addEventListener('icegatheringstatechange', checkState);
      }
    });
  }).then(function() {
    var offer = pc.localDescription;
    return offerRtcRequest(offer.sdp, offer.type);
  }).then(function(response) {
    console.log(response);
    return response.json();
  }).then(function(answer) {
    return pc.setRemoteDescription(answer);
  }).catch(function(e) {
    alert(e);
  });
}


function isMobile() {
    let check = false;
    (function(a){if(/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(a)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(a.substr(0,4))) check = true;})(navigator.userAgent||navigator.vendor||window.opera);
    return check;
};


export const constraints = {
  audio: {
    autoGainControl: false,
    sampleRate: 48000,
    sampleSize: 16,
    echoCancellation: true,
    noiseSuppression: true,
    channelCount: 1
  },
  video: isMobile()
};


export function start(pc, dc) {
  pc = createPeerConnection(pc);

  // add audio track
  navigator.mediaDevices.enumerateDevices()
    .then(function(devices) {
      const hasAudioInput = devices.find((device) => device.kind === "audioinput");
      var modifiedConstraints = {};
      modifiedConstraints.video = constraints.video;
      modifiedConstraints.audio = hasAudioInput ? constraints.audio : false;

      return Promise.resolve(modifiedConstraints);
    })
    .then(function(constraints) {
      if (constraints.audio || constraints.video) {
        return navigator.mediaDevices.getUserMedia(constraints);
      } else{
        return Promise.resolve(null);
      }
    })
    .then(function(stream) {
      if (stream) {
        stream.getTracks().forEach(function(track) {
          pc.addTrack(track, stream);
        });
      }

      return negotiate(pc);
    })
    .catch(function(err) {
      alert('Could not acquire media: ' + err);
    });

  var parameters = {"ordered": true};
  dc = pc.createDataChannel('data', parameters);
  dc.onclose = function() {
    clearInterval(controlCommandInterval);
    clearInterval(latencyInterval);
  };

  function sendJoystickOverDataChannel() {
    const {x, y} = getXY();
    var message = JSON.stringify({type: "testJoystick", data: {axes: [x, y], buttons: [false]}})
    dc.send(message);
  }
  function checkLatency() {
    const initialTime = new Date().getTime();
    pingHeadRequest().then(function() {
      const currentTime = new Date().getTime();
      if (Math.abs(currentTime - lastChannelMessageTime) < 1000) {
        const pingtime = currentTime - initialTime;
        pingPoints.push({'x': currentTime, 'y': pingtime});
        if (pingPoints.length > 1000) {
          pingPoints.shift();
        }
        chartPing.update();
        $("#ping-time").text((pingtime) + "ms");
      }
    })
  }
  dc.onopen = function() {
    controlCommandInterval = setInterval(sendJoystickOverDataChannel, 50);
    latencyInterval = setInterval(checkLatency, 1000);
    sendJoystickOverDataChannel();
  };

  const textDecoder = new TextDecoder();
  var carStaterIndex = 0;
  dc.onmessage = function(evt) {
    const text = textDecoder.decode(evt.data);
    const msg = JSON.parse(text);
    if (carStaterIndex % 100 == 0 && msg.type === 'carState') {
      const batteryLevel = Math.round(msg.data.fuelGauge * 100);
      $("#battery").text(batteryLevel + "%");
      batteryPoints.push({'x': new Date().getTime(), 'y': batteryLevel});
      if (batteryPoints.length > 1000) {
        batteryPoints.shift();
      }
      chartBattery.update();
    }
    carStaterIndex += 1;
    lastChannelMessageTime = new Date().getTime();
    $(".pre-blob").addClass('blob');
  };
}


export function stop(pc, dc) {
  if (dc) {
    dc.close();
  }
  if (pc.getTransceivers) {
    pc.getTransceivers().forEach(function(transceiver) {
      if (transceiver.stop) {
        transceiver.stop();
      }
    });
  }
  pc.getSenders().forEach(function(sender) {
    sender.track.stop();
  });
  setTimeout(function() {
    pc.close();
  }, 500);
}
