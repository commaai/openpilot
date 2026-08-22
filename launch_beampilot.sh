# fake car
export FINGERPRINT="HONDA_CIVIC_2022"
export SKIP_FW_QUERY="1"
export HSA_ENABLE_DXG_DETECTION=1

# IF it exists these are broken cpu remains
rm -f openpilot/selfdrive/modeld/models/*tinygrad.pkl*
rm -f openpilot/selfdrive/modeld/models/tg_input_devices.json

# wsl users use HIP (AMD) or 
export DEV="HIP"

# tici (c3 big) vs mici (c4 small)
# do 1 for tici, 0 for mici
export BIG="1"

echo "Compiling models for $DEV..."
scons -u -j$(nproc)

echo "Starting..."
exec ./launch_chffrplus.sh