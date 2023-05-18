// peer connection
var pc = null;

// data channel
var dc = null, dcInterval = null;

function createPeerConnection() {
    var config = {
        sdpSemantics: 'unified-plan'
    };

    pc = new RTCPeerConnection(config);

    // connect audio / video
    pc.addEventListener('track', function(evt) {
        console.log("here!")
        if (evt.track.kind == 'video')
            document.getElementById('video').srcObject = evt.streams[0];
        else
            document.getElementById('audio').srcObject = evt.streams[0];
    });

    return pc;
}

function negotiate() {
    return pc.createOffer({offerToReceiveAudio:true, offerToReceiveVideo:true}).then(function(offer) {
        return pc.setLocalDescription(offer);
    }).then(function() {
        return new Promise(function(resolve) {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
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
        return fetch('/offer', {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });
    }).then(function(response) {
        console.log(response);
        return response.json();
    }).then(function(answer) {
        return pc.setRemoteDescription(answer);
    }).catch(function(e) {
        alert(e);
    });
}

const keyVals = {w: 0, a: 0, s: 0, d: 0}
const handleKeyX = (event, setValue) => {
    const key = event.key.toLowerCase();
    if (['w', 'a', 's', 'd'].includes(key)){
        keyVals[key] = setValue;
        // console.log(keyVals);
    }
};

document.addEventListener('keydown', (e)=>(handleKeyX(e, 1)));
document.addEventListener('keyup', (e)=>(handleKeyX(e, 0)));

function getXY(){
    x = -keyVals.w + keyVals.s
    y = -keyVals.d + keyVals.a
    return {x, y}
}

function start() {
    pc = createPeerConnection();
    var parameters = {"ordered": true};

    dc = pc.createDataChannel('control_command', parameters);
    dc.onclose = function() {
        console.log("data channel closed");
        clearInterval(dcInterval);
    };
    dc.onopen = function() {
        dcInterval = setInterval(function() {
            const {x, y} = getXY();
            const dt = new Date().getTime();
            var message = JSON.stringify({x, y, dt});
            dc.send(message);
        }, 50);
    };
    dc.onmessage = function(evt) {
        // const times = JSON.parse(evt);
        console.log(evt);
    };

    var constraints = {
        audio: {
            autoGainControl: false,
            sampleRate: 48000,
            sampleSize: 16,
            echoCancellation: true,
            noiseSuppression: true,
            channelCount: 1
        },
        video: false
    };


    if (constraints.audio || constraints.video) {
        navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
            stream.getTracks().forEach(function(track) {
                console.log(track.getConstraints());
                console.log(track.getSettings());
                pc.addTrack(track, stream);
            });
            return negotiate();
        }, function(err) {
            alert('Could not acquire media: ' + err);
        });
    } else {
        negotiate();
    }
}

function stop() {
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
    // pc.getSenders().forEach(function(sender) {
    //     sender.track.stop();
    // });
    setTimeout(function() {
        pc.close();
    }, 500);
}

start();