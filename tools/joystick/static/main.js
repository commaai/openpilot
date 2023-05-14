var keyEnum = { W_Key:0, A_Key:1, S_Key:2, D_Key:3 };
var keyArray = new Array(4).fill(false);

const socket = io.connect('https://' + document.domain + ':' + location.port);

window.addEventListener('keydown', keydown_event);
window.addEventListener('keyup', keyup_event);

function keydown_event(e) {
	let key = '';
	switch (e.key) {
		case 'w':
                        keyArray[keyEnum.W_Key] = true;
                        break;
	}
	switch (e.key) {
		case 'a':
                        keyArray[keyEnum.A_Key] = true;
                        break;
	}
	switch (e.key) {
		case 's':
                        keyArray[keyEnum.S_Key] = true;
                        break;
	}
	switch (e.key) {
		case 'd':
                        keyArray[keyEnum.D_Key] = true;
                        break;
	}
}
function keyup_event(e) {
	let key = '';
	switch (e.key) {
		case 'w':
                        keyArray[keyEnum.W_Key] = false;
                        break;
	}
	switch (e.key) {
		case 'a':
                        keyArray[keyEnum.A_Key] = false;
                        break;
	}
	switch (e.key) {
		case 's':
                        keyArray[keyEnum.S_Key] = false;
                        break;
	}
	switch (e.key) {
		case 'd':
                        keyArray[keyEnum.D_Key] = false;
                        break;
	}
}

function isKeyDown(key)
{
    return keyArray[key];
}

setInterval(function(){
  var x = 0;
  var y = 0;
  if (isKeyDown(keyEnum.W_Key))
    x -= 1;
  if (isKeyDown(keyEnum.S_Key))
    x += 1;
  if (isKeyDown(keyEnum.D_Key))
    y -= 1;
  if (isKeyDown(keyEnum.A_Key))
    y += 1;
  socket.emit('control_command', {'x': x, 'y': y});

  document.getElementById("move_str").innerHTML = `Commanded motion: ${x}, ${y}`;

}, 50);

const audio = document.getElementById('audio-player');
let audioBuffer = [];
let audioQueue = [];

function int16ToBytes(value) {
    const bytes = new Uint8Array(2);
    bytes[0] = value & 0xff;
    bytes[1] = (value >> 8) & 0xff;
    return bytes;
}

function createWavFile(audioData) {
    const numChannels = 1;
    const sampleRate = 44100;
    const bitsPerSample = 16;
    const byteRate = (sampleRate * numChannels * bitsPerSample) / 8;
    const dataSize = audioData.length * (bitsPerSample / 8);

    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    // RIFF chunk descriptor
    view.setUint8(0, 'R'.charCodeAt(0));
    view.setUint8(1, 'I'.charCodeAt(0));
    view.setUint8(2, 'F'.charCodeAt(0));
    view.setUint8(3, 'F'.charCodeAt(0));
    view.setUint32(4, 36 + dataSize, true);
    view.setUint8(8, 'W'.charCodeAt(0));
    view.setUint8(9, 'A'.charCodeAt(0));
    view.setUint8(10, 'V'.charCodeAt(0));
    view.setUint8(11, 'E'.charCodeAt(0));

    // FMT sub-chunk
    view.setUint8(12, 'f'.charCodeAt(0));
    view.setUint8(13, 'm'.charCodeAt(0));
    view.setUint8(14, 't'.charCodeAt(0));
    view.setUint8(15, ' '.charCodeAt(0));
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, numChannels * (bitsPerSample / 8), true);
    view.setUint16(34, bitsPerSample, true);
    event
    // Data sub-chunk
    view.setUint8(36, 'd'.charCodeAt(0));
    view.setUint8(37, 'a'.charCodeAt(0));
    view.setUint8(38, 't'.charCodeAt(0));
    view.setUint8(39, 'a'.charCodeAt(0));
    view.setUint32(40, dataSize, true);
    for (let i = 0; i < audioData.length; i++) {
        const data = int16ToBytes(audioData[i]);
        view.setUint8(44 + i * 2, data[0]);
        view.setUint8(44 + i * 2 + 1, data[1]);
    }

    return new Blob([view], { type: 'audio/wav' });
}

function playAudioData() {
    const wavFile = createWavFile(audioBuffer);
    audio.src = URL.createObjectURL(wavFile);
    audio.play();
    audioBuffer = [];
}

socket.on('stream', (audioData) => {
    audioBuffer = audioBuffer.concat(audioData);
    if (audioBuffer.length >= 44100) {
        playAudioData();
    }
});

socket.on('battery', (batter_fraction) => {
    var battery_perc = batter_fraction * 100;
    document.getElementById("robot_state").innerHTML = `Commanded motion: ${battery_perc} %`;

});

var startRecordingButton = document.getElementById("startRecordingButton");
var leftchannel = [];
var rightchannel = [];
var recorder = null;
var recordingLength = 0;
var volume = null;
var mediaStream = null;
var sampleRate = 44100;
var context = null;

// Initialize recorder
navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;

function success(e) {
    console.log("user consent");

    // creates the audio context
    window.AudioContext = window.AudioContext || window.webkitAudioContext;
    context = new AudioContext();

    // creates an audio node from the microphone incoming stream
    mediaStream = context.createMediaStreamSource(e);

    // https://developer.mozilla.org/en-US/docs/Web/API/AudioContext/createScriptProcessor
    // bufferSize: the onaudioprocess event is called when the buffer is full
    var bufferSize = 16384;
    var numberOfInputChannels = 1;
    var numberOfOutputChannels = 2;
    if (context.createScriptProcessor) {
        recorder = context.createScriptProcessor(bufferSize, numberOfInputChannels, numberOfOutputChannels);
    } else {
        recorder = context.createJavaScriptNode(bufferSize, numberOfInputChannels, numberOfOutputChannels);
    }

    recorder.onaudioprocess = function (e) {
        var arrr = new Float32Array(e.inputBuffer.getChannelData(0))
        console.log(arrr)
        socket.emit('audio_blob', arrr);
    }

    // we connect the recorder
    mediaStream.connect(recorder);
    recorder.connect(context.destination);
}

function error(e) {
    console.error(e);
}

navigator.getUserMedia({audio: true}, success, error);
