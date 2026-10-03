# Showcase

Despite being a tiny library, tinygrad is capable of doing a lot of things. From state-of-the-art [vision](https://arxiv.org/abs/1905.11946) to state-of-the-art [language](https://arxiv.org/abs/1706.03762) models.

## Vision

### EfficientNet

You can either pass in the URL of a picture to discover what it is:
```sh
python3 examples/efficientnet.py ./test/models/efficientnet/Chicken.jpg
```
Or, if you have a camera and OpenCV installed, you can detect what is in front of you:
```sh
python3 examples/efficientnet.py webcam
```

### YOLOv8

Take a look at [yolov8.py](https://github.com/tinygrad/tinygrad/tree/master/examples/yolov8.py).

![yolov8 by tinygrad](https://github.com/tinygrad/tinygrad/blob/master/docs/showcase/yolov8_showcase_image.png?raw=true)

## Audio

### Whisper

Take a look at [whisper.py](https://github.com/tinygrad/tinygrad/tree/master/examples/whisper.py). You need pyaudio and torchaudio installed.

```sh
SMALL=1 python3 examples/whisper.py
```

## Generative

### Stable Diffusion

```sh
python3 examples/stable_diffusion.py
```

![a horse sized cat eating a bagel](https://github.com/tinygrad/tinygrad/blob/master/docs/showcase/stable_diffusion_by_tinygrad.jpg?raw=true)

*"a horse sized cat eating a bagel"*

### LLaMA

You will need to download and put the weights into the `weights/LLaMA` directory, which may need to be created.

Then you can have a chat with Stacy:
```sh
python3 examples/llama.py
```

### Conversation

Make sure you have espeak installed and `PHONEMIZER_ESPEAK_LIBRARY` set.

Then you can talk to Stacy:
```sh
python3 examples/conversation.py
```
