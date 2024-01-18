The openwakeword driectory code was copied from https://github.com/dscripka/openWakeWord 
and then stripped down to the essentials for Openpilot's purposes.

To test wake word detection on the comma device or PC, run wakeword.py and say "alexa".
Make sure you have onnxruntime==1.16.3 when running on the comma device.

pip install onnxruntime==1.16.3

You can also run rev_speechd.py which will wait for the "WakeWordDetected" param to be set.
To setup the Rev.Ai api you need to install rev_ai:

pip install rev_ai

You also need to set your rev ai acccess token which can be obtained with a free account. https://www.rev.ai/access-token
Once you have your token you can paste it in launch_openpilot.sh.

export REVAI_ACCESS_TOKEN=""

Once you have everything set up you can run ./launch_openpilot and see the assistant overlay on the UI. You can also run ./ui, rev_speechd.py, wakeword.py, micd.py in their own terminals for testing.
