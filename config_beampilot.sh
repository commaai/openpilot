# this is a CONFIG FILE
# launch_beampilot will source this file

# fake car
# if you change this, you have to fix beamngd
# and possibly beamcamd too as camera positions
# and CAN formats differ from car to car
export FINGERPRINT="HONDA_CIVIC_2022"
export SKIP_FW_QUERY="1"
export HSA_ENABLE_DXG_DETECTION=1

# to use gpu, pick one, (AMD) or (NV)idia
export USE_AMD=1
# export USE_NV=1

# tici (c3 big) vs mici (c4 small)
# do 1 for tici, 0 for mici
export BIG="1"

# chestnut class model selection
# chestnut is the eGPU line of models from comma
# for comma hardware; it can run on desktop with enough resources
# anyone without a strong dedicated dGPU should use non-chestnut
# see more about it in the readme or online at comma.ai in a blogpost somewhere
export CHESTNUT="1"