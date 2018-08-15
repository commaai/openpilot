import os
import subprocess
import sys

"call the snd library"""
base_fld_code = "/data/openpilot/selfdrive/car/modules/snd/"
base_fld_wav = "/data/openpilot/selfdrive/car/modules/snd/"
snd_command = base_fld_wav
snd_beep = int(sys.argv[1])
gen_snd = True
if (snd_beep == 0): #no beep
    gen_snd = False
elif (snd_beep == 2):
  snd_command += "enable.wav"
elif (snd_beep == 1):
  snd_command += "disable.wav"
elif (snd_beep == 4):
  snd_command +="attention.wav"
elif (snd_beep == 3):
  snd_command += "info.wav"
else:
  snd_command += "error.wav"
if (gen_snd):
  env = dict(os.environ)
  env['LD_LIBRARY_PATH'] = base_fld_code + "."
  args = [base_fld_code + "mediaplayer", snd_command]
  subprocess.Popen(args, shell = False, stdin=None, stdout=None, stderr=None, env = env, close_fds=True)
