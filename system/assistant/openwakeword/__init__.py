import os
from openpilot.system.assistant.openwakeword.model import Model
assert Model

FEATURE_MODELS = {
  "embedding": {
    "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/embedding_model.tflite"),
    "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite"
  },
  "melspectrogram": {
    "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/melspectrogram.tflite"),
    "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite"
  }
}

VAD_MODELS = {
  "silero_vad": {
    "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/silero_vad.onnx"),
    "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/silero_vad.onnx"
  }
}

MODELS = {
  "alexa": {
    "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/alexa_v0.1.tflite"),
    "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/alexa_v0.1.tflite"
  },
  "hey_mycroft": {
    "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/hey_mycroft_v0.1.tflite"),
    "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_mycroft_v0.1.tflite"
  },
  "hey_jarvis": {
    "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/hey_jarvis_v0.1.tflite"),
    "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.tflite"
  },
  "hey_rhasspy": {
    "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/hey_rhasspy_v0.1.tflite"),
    "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_rhasspy_v0.1.tflite"
  },
  "timer": {
    "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/timer_v0.1.tflite"),
    "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/timer_v0.1.tflite"
  },
  "weather": {
    "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/models/weather_v0.1.tflite"),
    "download_url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/weather_v0.1.tflite"
  }
}


