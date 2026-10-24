# alfred sources this at the top of every CI step, on the UP and on the DUT
source "$WORKSPACE/launch_env.sh"
export LD_LIBRARY_PATH="$(python -c 'import ffmpeg; print(ffmpeg.LIB_DIR)' 2>/dev/null):/usr/local/lib:${LD_LIBRARY_PATH:-}"
export GIT_SSH_COMMAND='ssh -i /data/gitkey'
ln -snf "$WORKSPACE" /data/pythonpath
