// peer connection
var pc = null;

// data channel
var dc = null, dcInterval = null;
var battery = null;

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
const handleKeyX = (key, setValue) => {
    if (['w', 'a', 's', 'd'].includes(key)){
        keyVals[key] = setValue;
        let color = "#333";
        if (setValue === 1){
            color = "#e74c3c";

        }
        $("#key-"+key).css('background', color);
        const {x, y} = getXY()
        $("#pos-vals").text(x+","+y)
    }
};

document.addEventListener('keydown', (e)=>(handleKeyX(e.key.toLowerCase(), 1)));
document.addEventListener('keyup', (e)=>(handleKeyX(e.key.toLowerCase(), 0)));
$(".keys").mousedown((e)=>{
    handleKeyX($(e.target).attr('id').replace('key-', ''), 1);
})
$(".keys").mouseup((e)=>{
    handleKeyX($(e.target).attr('id').replace('key-', ''), 0);
})

function getXY(){
    x = -keyVals.w + keyVals.s
    y = -keyVals.d + keyVals.a
    return {x, y}
}

function start() {
    pc = createPeerConnection();

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

    var parameters = {"ordered": true};
    dc = pc.createDataChannel('data', parameters);
    dc.onclose = function() {
        console.log("data channel closed");
        clearInterval(dcInterval);
        clearInterval(batteryInterval);
    };
    dc.onopen = function() {
        dcInterval = setInterval(function() {
            const {x, y} = getXY();
            const dt = new Date().getTime();
            var message = JSON.stringify({type: 'control_command', x, y, dt});
            dc.send(message);
        }, 50);
        batteryInterval = setInterval(function() {
            var message = JSON.stringify({type: 'battery_level'});
            dc.send(message);
        }, 10000);
    };
    let val_print_idx = 0;
    dc.onmessage = function(evt) {
        const data = JSON.parse(evt.data);
        console.log(data);
        if(val_print_idx == 0 && data.type === 'ping_time'){
            $("#ping-time").text((data.outgoing_time - data.incoming_time) + "ms");
        }
        val_print_idx = (val_print_idx + 1 ) % 20;
        if(data.type === 'battery_level'){
            $("#battery").text(data.value + "%");
        }
        
    };
    $(".pre-blob").addClass('blob');

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