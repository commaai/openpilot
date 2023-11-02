#!/usr/bin/bash

# force CPU rendering on respective platforms
if [ -f /TICI ]; then
  export QT_QPA_PLATFORM=eglfs                                    # offscreen doesn't work with EGL/GLES
  export LP_NUM_THREADS=0                                         # disable threading so we stay on our assigned CPU
else
  export QT_QPA_PLATFORM=offscreen
  export __GLX_VENDOR_LIBRARY_NAME=mesa                           # for PC with libglvnd, should not effect other platforms
fi