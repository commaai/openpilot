// peer connection
var pc = null;

// data channel
var dc = null, dcInterval = null;
var battery = null;
var last_ping = null;

function createPeerConnection() {
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
        const {x, y} = getXY();
        $("#pos-vals").text(x+","+y);
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
    function controlCommand() {
        const {x, y} = getXY();
        const dt = new Date().getTime();
        var message = JSON.stringify({type: 'control_command', x, y, dt});
        dc.send(message);
    }

    function batteryLevel() {
        var message = JSON.stringify({type: 'battery_level'});
        dc.send(message);
    }
    
    dc.onopen = function() {
        dcInterval = setInterval(controlCommand, 50);
        batteryInterval = setInterval(batteryLevel, 60000);
        controlCommand();
        batteryLevel();
        $(".sound").click((e)=>{
            const sound = $(e.target).attr('id').replace('sound-', '')
            dc.send(JSON.stringify({type: 'play_sound', sound}));;
        })
    };
    let val_print_idx = 0;
    dc.onmessage = function(evt) {
        const data = JSON.parse(evt.data);
        // console.log(data);
        if(val_print_idx == 0 && data.type === 'ping_time'){
            const dt = new Date().getTime();
            $("#ping-time").text((dt - data.incoming_time) + "ms");
            last_ping = dt;
            $(".pre-blob").addClass('blob');
        }
        val_print_idx = (val_print_idx + 1 ) % 20;
        if(data.type === 'battery_level'){
            $("#battery").text(data.value + "%");
        }
        
    };
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
    pc.getSenders().forEach(function(sender) {
        sender.track.stop();
    });
    setTimeout(function() {
        pc.close();
    }, 500);
}

setInterval(()=>{
    const dt = new Date().getTime();
    if ((dt - last_ping) > 1000){
        $(".pre-blob").removeClass('blob');
        $("#battery").text("-");
        $("#ping-time").text('-');
        $("video")[0].load();
    }
}, 5000);

const sleep = ms => new Promise(r => setTimeout(r, ms));
$("#plan-button").click(async function(){
    let plan = $("#plan-text").val();
    const planList = []
    plan.split("\n").forEach(function(e){
        let line = e.split(",").map(k=>parseInt(k));
        if (line.length != 5 || line.slice(0, 4).map(e=>[1, 0].includes(e)).includes(false) || line[4] < 0 || line[4] > 10){
            console.log("invalid plan");
        }
        else{
            planList.push(line)
        }
    });
    async function execute () {
        for (var i = 0; i < planList.length; i++) {
            let [w, a, s, d, t] = planList[i];
            while(t > 0){
                console.log(w, a, s, d, t);
                if(w==1){$("#key-w").mousedown();}
                if(a==1){$("#key-a").mousedown();}
                if(s==1){$("#key-s").mousedown();}
                if(d==1){$("#key-d").mousedown();}
                await sleep(50);
                $("#key-w").mouseup();
                $("#key-a").mouseup();
                $("#key-s").mouseup();
                $("#key-d").mouseup();
                t = t - 0.05;
            }
        }
      }
    execute();
});


// start();