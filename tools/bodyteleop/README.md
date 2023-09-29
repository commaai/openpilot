# bodyteleop

## Components
- `web.py` is the tele-operation server that starts automatically when the commabody goes onroad, which can be found at `https://<body-ip>:5000`.
- `bodyav.py` has all the audio/video webRTC tracks
- `static/` contains the teleop ui
- `bodycontrolsd.py` gets all relevant input messages, processes them and sends the final `testJoystick` message (which the body executes).

For hackathon instructions check out [HACKATHON.md](./HACKATHON.md)