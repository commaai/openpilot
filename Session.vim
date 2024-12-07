let SessionLoad = 1
let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd ~/Documents/projects/drone/openpilot
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
let s:shortmess_save = &shortmess
if &shortmess =~ 'A'
  set shortmess=aoOA
else
  set shortmess=aoO
endif
badd +1 ~/Documents/projects/drone/openpilot
badd +90 tools/sim/bridge/gz/gz_world.py
badd +38 tools/sim/bridge/metadrive/metadrive_world.py
badd +20 tools/sim/lib/common.py
badd +1 tools/sim/lib
badd +36 tools/sim/lib/camerad.py
badd +106 tools/sim/lib/simulated_sensors.py
badd +1 tools/sim/bridge/metadrive
badd +144 tools/sim/bridge/metadrive/metadrive_process.py
badd +86 ~/PX4-Autopilot/src/modules/simulation/gz_bridge/GZBridge.cpp
badd +85 tools/sim/bridge/metadrive/metadrive_bridge.py
badd +19 tools/sim/bridge/common.py
badd +102 tools/sim/lib/simulated_car.py
badd +1 tools/sim/bridge/gz/ros_ws/src/rosserver/rosserver/__init__.py
badd +52 tools/sim/bridge/gz/ros_ws/src/rosserver/rosserver/server.py
badd +23 tools/sim/bridge/gz/ros_ws/src/rosserver/setup.py
badd +1 tools/sim/bridge/gz/ros_ws/src/rosserver/package.xml
badd +19 tools/sim/bridge/gz/msgs.py
badd +1 tools/sim/bridge/gz/__init__.py
badd +1 tools/sim/bridge/__init__.py
badd +236 selfdrive/locationd/calibrationd.py
badd +1 docs/concepts/logs.md
badd +1 selfdrive/locationd
badd +99 selfdrive/locationd/torqued.py
badd +1 selfdrive/locationd/paramsd.py
badd +1 selfdrive/locationd/models/pose_kf.py
badd +1 selfdrive/locationd/models
badd +41 selfdrive/locationd/models/car_kf.py
badd +11 ~/Documents/projects/drone/ros2_ws/src/offboard_csv/setup.py
badd +94 ~/Documents/projects/drone/ros2_ws/src/offboard_csv/offboard_csv/offboard_csv.py
badd +1 .dockerignore
badd +1 tools/sim/bridge/gz/ros_ws/src/rosserver/rosserver
badd +43 ~/.local/share/nvim/mason/packages/jedi-language-server/venv/lib/python3.10/site-packages/jedi/third_party/typeshed/stdlib/3/multiprocessing/__init__.pyi
badd +10 tools/sim/bridge/gz/gz_bridge.py
badd +17 tools/sim/launch_px4_sitl.sh
badd +571 ~/PX4-Autopilot/Tools/simulation/gz/worlds/road.sdf
badd +1 ~/PX4-Autopilot/Tools/simulation/gz/worlds
badd +14 system/camerad/main.cc
badd +1 system/camerad
badd +204 system/camerad/cameras/process_raw.cl
badd +1 system/camerad/cameras
badd +51 system/camerad/cameras/camera_common.cc
badd +26 system/camerad/cameras/camera_common.h
badd +295 system/camerad/cameras/camera_qcom2.cc
badd +14 msgq_repo/msgq/visionipc/visionipc_server.h
badd +1 msgq_repo/msgq/visionipc
badd +42 msgq_repo/msgq/visionipc/visionipc_server.cc
badd +164 system/camerad/cameras/spectra.h
badd +84 system/camerad/cameras/spectra.cc
badd +211 system/manager/manager.py
badd +291 system/manager/process.py
badd +1 system/manager/helpers.py
badd +22 system/manager/build.py
badd +70 system/manager/process_config.py
badd +1 system/camerad/__init__.py
badd +1 system/camerad/cameras/cdm.cc
badd +1 tools/sim/launch_openpilot.sh
badd +1 tools/sim
badd +1 tools/sim/run_bridge.py
badd +17 launch_chffrplus.sh
badd +8 launch_env.sh
badd +165 selfdrive/pandad/pandad.py
badd +1 selfdrive/pandad/pandad
badd +226 selfdrive/pandad/pandad.cc
badd +26 selfdrive/pandad/panda.cc
badd +308 selfdrive/pandad/spi.cc
badd +20 selfdrive/pandad/main.cc
badd +11 system/ugpsd.py
badd +42 cereal/messaging/__init__.py
badd +1 cereal/messaging
badd +90 cereal/messaging/bridge.cc
badd +54 cereal/messaging/messaging.h
badd +14 selfdrive/pandad/SConscript
badd +34 selfdrive/pandad/fakepanda.cc
badd +5946 cereal/gen/cpp/log.capnp.c++
badd +1 cereal/gen/cpp
badd +1 cereal/custom.capnp
badd +1 cereal
badd +32 cereal/services.py
badd +29 msgq_repo/msgq/ipc.cc
badd +1 msgq_repo/msgq
badd +6 msgq_repo/msgq/event.h
badd +1 msgq_repo/msgq/msgq
badd +1 msgq_repo/msgq/ipc_pyx.o
badd +1 msgq_repo/msgq/ipc.h
badd +31 msgq_repo/msgq/ipc.pxd
badd +58 msgq_repo/msgq/event.cc
badd +366 SConstruct
badd +310 cereal/log.capnp
badd +8 cereal/gen/cpp/car.capnp.c++
badd +171 cereal/gen/cpp/car.capnp.h
badd +232 system/hardware/hardwared.py
badd +1 system/hardware
badd +217 cereal/gen/cpp/log.capnp.h
badd +2 selfdrive/SConscript
badd +116 system/version.py
badd +1 common/version.h
badd +1 common
badd +6 system/updated/casync/common.py
badd +1 system
badd +7 common/git.py
badd +28 RELEASES.md
badd +11 selfdrive/controls/controlsd.py
badd +90 selfdrive/car/card.py
badd +11 selfdrive/car/car_helpers.py
badd +1 selfdrive/car/vin.py
badd +1 selfdrive/car/isotp_parallel_query.py
badd +173 msgq_repo/msgq/tests/test_fake.py
badd +1 selfdrive/pandad/panda_comms.cc
badd +15 selfdrive/pandad/panda.h
badd +468 selfdrive/test/process_replay/process_replay.py
badd +1 selfdrive/ui/ui.h
badd +1 selfdrive/ui
badd +358 selfdrive/ui/ui.cc
badd +48 selfdrive/ui/ui.py
badd +1 selfdrive/ui/ui
badd +15 selfdrive/ui/main.cc
badd +401 selfdrive/ui/qt/offroad/settings.cc
badd +1 selfdrive/ui/qt/offroad
badd +20 selfdrive/ui/qt/offroad/software_settings.cc
badd +6856 selfdrive/pandad/pandad_api_impl.cpp
badd +93 selfdrive/pandad/panda_comms.h
badd +45 panda/board/health.h
badd +178 panda/board/obj/panda.bin.signed
badd +1 panda/board/obj
badd +184 panda/board/obj/panda_h7.bin.signed
badd +1 panda/board/obj/version
badd +83 common/util.cc
badd +111 selfdrive/test/test_onroad.py
badd +209 selfdrive/car/fw_versions.py
badd +56 selfdrive/car/fw_query_definitions.py
badd +102 panda/python/uds.py
badd +4 selfdrive/pandad/can_list_to_can_capnp.cc
badd +26 opendbc_repo/opendbc/can/tests/test_packer_parser.py
badd +6 panda/board/can.h
badd +71 system/loggerd/loggerd.cc
badd +32 common/i2c.cc
badd +52 msgq_repo/msgq/__init__.py
badd +43 selfdrive/car/ecu_addrs.py
badd +1 selfdrive/car/ford/interface.py
badd +9 selfdrive/car/subaru/interface.py
badd +240 selfdrive/car/interfaces.py
badd +14 selfdrive/car/volkswagen/interface.py
badd +1 selfdrive/car/volkswagen
badd +1 selfdrive/car/volkswagen/carstate.py
badd +1 selfdrive/car/body/carstate.py
badd +1 selfdrive/car/body
badd +22 selfdrive/car/body/values.py
badd +1 selfdrive/car/body/bodycan.py
badd +78 selfdrive/car/body/carcontroller.py
badd +1 selfdrive/car/body/fingerprints.py
badd +1 selfdrive/car/body/interface.py
badd +1 selfdrive/car/body/radar_interface.py
badd +1 selfdrive/car
badd +1 selfdrive/car/body/__init__.py
badd +42 selfdrive/controls/lib/pid.py
badd +72 ~/Documents/projects/deportivo-morelos/backend/prisma/seed.ts
badd +32 ~/Documents/projects/deportivo-morelos/admin/src/pages/home/index.tsx
badd +1 ~/Documents/projects/drone-env-setup/main.py
badd +105 selfdrive/car/nissan/values.py
badd +1 selfdrive/car/nissan
badd +19 selfdrive/car/nissan/fingerprints.py
badd +11 body/board/comms.h
badd +1 body/board
badd +6 body/board/flash_base.sh
badd +1 body/board/flash_knee.sh
badd +59 body/board/canloader.py
badd +18 opendbc_repo/opendbc/car/body/fingerprints.py
badd +1 opendbc_repo/opendbc/car/body
badd +1 opendbc_repo/opendbc/car/body/radar_interface.py
badd +1 opendbc_repo/opendbc/car/body/values.py
badd +8 selfdrive/car/CARS_template.md
badd +16 selfdrive/car/README.md
badd +80 opendbc_repo/opendbc/SConstruct
badd +15 opendbc_repo/opendbc/comma_body.dbc
badd +22 opendbc_repo/opendbc/car/docs.py
badd +17 opendbc_repo/SConstruct
badd +4 common/basedir.py
badd +1 selfdrive/car/drone/__init__.py
badd +1 selfdrive/car/drone
badd +1 selfdrive/car/drone/bodycan.py
badd +7 selfdrive/car/drone/carcontroller.py
badd +18 selfdrive/car/drone/fingerprints.py
badd +21 selfdrive/car/drone/values.py
badd +117 opendbc_repo/opendbc/car/car_helpers.py
badd +1 opendbc_repo/opendbc/dbc/comma_body.dbc
badd +26 selfdrive/car/__init__.py
badd +19 opendbc_repo/opendbc/can/parser_pyx.pyx
badd +1 opendbc_repo/opendbc/can/parser.cc
badd +1 opendbc_repo/opendbc/can/parser.py
badd +31 opendbc_repo/opendbc/can/common_dbc.h
badd +80 opendbc_repo/opendbc/can/common.h
badd +1 opendbc_repo/opendbc/can/common.cc
badd +239 opendbc_repo/opendbc/can/dbc.cc
badd +31 selfdrive/car/drone/carstate.py
badd +7 selfdrive/car/drone/interface.py
badd +1 selfdrive/controls
badd +8 cereal/__init__.py
badd +677 cereal/car.capnp
badd +78 selfdrive/car/chrysler/carcontroller.py
badd +5 opendbc_repo/opendbc/drone.dbc
badd +28 opendbc_repo/opendbc/ford_fusion_2018_adas.dbc
badd +1 panda/board/safety/safety_body.h
badd +1 panda/board/safety
badd +25 panda/board/safety/safety_nissan.h
badd +5 opendbc_repo/opendbc/nissan_leaf_2018_generated.dbc
badd +32 panda/board/safety/safety_drone.h
badd +145 panda/board/safety_declarations.h
badd +1 panda/board
badd +52 panda/board/safety.h
badd +1 opendbc_repo/opendbc/requirements.txt
badd +1 selfdrive/car/drone/radar_interface.py
badd +1 selfdrive/pandad
badd +1 selfdrive
badd +49 term://~/Documents/projects/deportivo-morelos/admin//205843:/usr/bin/zsh
badd +26 system/camerad/cameras/camera_util.cc
badd +94 system/qcomgpsd/qcomgpsd.py
badd +1 opendbc
badd +1 selfdrive/car/torque_data/substitute.toml
badd +1 selfdrive/car/torque_data
badd +1 selfdrive/car/torque_data/params.toml
badd +37 selfdrive/car/torque_data/override.toml
badd +1 selfdrive/car/torque_data/neural_ff_weights.json
badd +17 selfdrive/car/values.py
badd +349 selfdrive/car/docs_definitions.py
badd +2 selfdrive/car/fingerprints.py
badd +46 opendbc_repo/opendbc/can/packer.cc
badd +1 opendbc_repo/opendbc/can
badd +7 term://~/Documents/projects/drone/openpilot//126415:/usr/bin/zsh
badd +113 panda/SConscript
badd +1 system/loggerd/encoderd.cc
badd +1 tools/sim/README.md
badd +27 .devcontainer/devcontainer.json
badd +1 .devcontainer
badd +13 .devcontainer/Dockerfile
badd +1 tools/sim/bridge
badd +1 tools/sim/bridge/gz
badd +1 tools/sim/bridge/gz/ros_ws/src/px4_msgs/msg/MavlinkLog.msg
badd +1 tools/sim/bridge/gz/ros_ws/src/px4_msgs/msg
badd +1 tools/sim/bridge/gz/mavlink
badd +30 tools/sim/bridge/gz/mavlink/CMakeLists.txt
badd +1 tools/sim/bridge/gz/mavlink/mavlink-bridge.cpp
badd +225 tools/op.sh
badd +1 tools
badd +3 .devcontainer/.gitignore
badd +38 .devcontainer/container_post_create.sh
badd +15 .devcontainer/container_post_start.sh
badd +46 .devcontainer/host_setup
badd +9 .devcontainer/host_setup.cmd
badd +7 tools/sim/run_gz_bridge.py
badd +194 /usr/include/netinet/in.h
badd +1 tools/sim/bridge/gz/mavlink/find_package(MAVSDK\ REQUIRED)
badd +1 tools/sim/bridge/gz/mavlink/\#\ Finds\ MAVSDK\ when\ installed\ system\ wide.
badd +1 tools/sim/bridge/gz/mavlink/utils.hpp
badd +22 tools/sim/bridge/gz/mavlink/mavlink-bridge.hpp
badd +27 /usr/include/gz/msgs10/gz/msgs.hh
badd +12 /usr/include/gz/msgs10/gz/msgs/image.pb.h
badd +253 /usr/include/gz/msgs10/gz/msgs/details/image.pb.h
badd +1 cereal/include/c++.capnp
badd +1 cereal/include
argglobal
%argdel
$argadd ~/Documents/projects/drone/openpilot
set stal=2
tabnew +setlocal\ bufhidden=wipe
tabnew +setlocal\ bufhidden=wipe
tabrewind
edit tools/sim/bridge/gz/mavlink/mavlink-bridge.cpp
let s:save_splitbelow = &splitbelow
let s:save_splitright = &splitright
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
wincmd _ | wincmd |
split
1wincmd k
wincmd w
let &splitbelow = s:save_splitbelow
let &splitright = s:save_splitright
wincmd t
let s:save_winminheight = &winminheight
let s:save_winminwidth = &winminwidth
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 104 + 105) / 210)
exe '2resize ' . ((&lines * 24 + 26) / 52)
exe 'vert 2resize ' . ((&columns * 105 + 105) / 210)
exe '3resize ' . ((&lines * 24 + 26) / 52)
exe 'vert 3resize ' . ((&columns * 105 + 105) / 210)
argglobal
balt /usr/include/netinet/in.h
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 101 - ((31 * winheight(0) + 24) / 49)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 101
normal! 0
lcd ~/Documents/projects/drone/openpilot
wincmd w
argglobal
if bufexists(fnamemodify("~/Documents/projects/drone/openpilot/tools/sim/bridge/gz/mavlink/mavlink-bridge.hpp", ":p")) | buffer ~/Documents/projects/drone/openpilot/tools/sim/bridge/gz/mavlink/mavlink-bridge.hpp | else | edit ~/Documents/projects/drone/openpilot/tools/sim/bridge/gz/mavlink/mavlink-bridge.hpp | endif
if &buftype ==# 'terminal'
  silent file ~/Documents/projects/drone/openpilot/tools/sim/bridge/gz/mavlink/mavlink-bridge.hpp
endif
balt ~/Documents/projects/drone/openpilot/tools/sim/bridge/gz/mavlink/mavlink-bridge.cpp
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal nofen
silent! normal! zE
let &fdl = &fdl
let s:l = 15 - ((9 * winheight(0) + 12) / 24)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 15
normal! 09|
lcd ~/Documents/projects/drone/openpilot
wincmd w
argglobal
if bufexists(fnamemodify("~/Documents/projects/drone/openpilot/tools/sim/bridge/gz/mavlink/mavlink-bridge.cpp", ":p")) | buffer ~/Documents/projects/drone/openpilot/tools/sim/bridge/gz/mavlink/mavlink-bridge.cpp | else | edit ~/Documents/projects/drone/openpilot/tools/sim/bridge/gz/mavlink/mavlink-bridge.cpp | endif
if &buftype ==# 'terminal'
  silent file ~/Documents/projects/drone/openpilot/tools/sim/bridge/gz/mavlink/mavlink-bridge.cpp
endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 230 - ((8 * winheight(0) + 12) / 24)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 230
normal! 0
lcd ~/Documents/projects/drone/openpilot
wincmd w
exe 'vert 1resize ' . ((&columns * 104 + 105) / 210)
exe '2resize ' . ((&lines * 24 + 26) / 52)
exe 'vert 2resize ' . ((&columns * 105 + 105) / 210)
exe '3resize ' . ((&lines * 24 + 26) / 52)
exe 'vert 3resize ' . ((&columns * 105 + 105) / 210)
tabnext
edit ~/Documents/projects/drone/openpilot/cereal/log.capnp
let s:save_splitbelow = &splitbelow
let s:save_splitright = &splitright
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
let &splitbelow = s:save_splitbelow
let &splitright = s:save_splitright
wincmd t
let s:save_winminheight = &winminheight
let s:save_winminwidth = &winminwidth
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
wincmd =
argglobal
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 310 - ((25 * winheight(0) + 24) / 49)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 310
normal! 016|
lcd ~/Documents/projects/drone/openpilot
wincmd w
argglobal
if bufexists(fnamemodify("~/Documents/projects/drone/openpilot/cereal/car.capnp", ":p")) | buffer ~/Documents/projects/drone/openpilot/cereal/car.capnp | else | edit ~/Documents/projects/drone/openpilot/cereal/car.capnp | endif
if &buftype ==# 'terminal'
  silent file ~/Documents/projects/drone/openpilot/cereal/car.capnp
endif
balt ~/Documents/projects/drone/openpilot/cereal/log.capnp
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 188 - ((38 * winheight(0) + 24) / 49)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 188
normal! 08|
lcd ~/Documents/projects/drone/openpilot
wincmd w
2wincmd w
wincmd =
tabnext
edit ~/Documents/projects/drone/openpilot/tools/sim/run_bridge.py
let s:save_splitbelow = &splitbelow
let s:save_splitright = &splitright
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
let &splitbelow = s:save_splitbelow
let &splitright = s:save_splitright
wincmd t
let s:save_winminheight = &winminheight
let s:save_winminwidth = &winminwidth
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
wincmd =
argglobal
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 11 - ((10 * winheight(0) + 24) / 49)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 11
normal! 0
lcd ~/Documents/projects/drone/openpilot
wincmd w
argglobal
if bufexists(fnamemodify("~/Documents/projects/drone/openpilot/.devcontainer/devcontainer.json", ":p")) | buffer ~/Documents/projects/drone/openpilot/.devcontainer/devcontainer.json | else | edit ~/Documents/projects/drone/openpilot/.devcontainer/devcontainer.json | endif
if &buftype ==# 'terminal'
  silent file ~/Documents/projects/drone/openpilot/.devcontainer/devcontainer.json
endif
balt ~/Documents/projects/drone/openpilot/tools/sim/run_bridge.py
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 27 - ((24 * winheight(0) + 24) / 49)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 27
normal! 02|
lcd ~/Documents/projects/drone/openpilot
wincmd w
wincmd =
tabnext 2
set stal=1
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0 && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20
let &shortmess = s:shortmess_save
let &winminheight = s:save_winminheight
let &winminwidth = s:save_winminwidth
let s:sx = expand("<sfile>:p:r")."x.vim"
if filereadable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &g:so = s:so_save | let &g:siso = s:siso_save
nohlsearch
let g:this_session = v:this_session
let g:this_obsession = v:this_session
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
