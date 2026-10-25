# alfred sources this at the top of every CI step, on the UP and on the DUT
source "$WORKSPACE/launch_env.sh"
export GIT_SSH_COMMAND='ssh -i /data/gitkey'
if [ -f /AGNOS ]; then
  export LD_LIBRARY_PATH="$(python -c 'import ffmpeg; print(ffmpeg.LIB_DIR)'):/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  ln -snf "$WORKSPACE" /data/pythonpath
fi
