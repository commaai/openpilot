# PC에서 웹캠으로 openpilot 실행

필요 항목:
- Ubuntu 20.04
- GPU (권장사항)
- 최소 720p 화소 및 78도 이상의 시야 기능이 있는 2개의 웹캠 (예시: 로지텍 C920/C615)
- 차량에 연결할 black panda와 [자동차 하네스](https://comma.ai/shop/products/comma-car-harness)
- panda를 컴퓨터에 연결하기 위한 [Panda paw](https://comma.ai/shop/products/panda-paw) 또는 USB-A to USB-A 케이블

## openpilot 초기 설정
```
cd ~
git clone https://github.com/commaai/openpilot.git
```
- [이 readme](https://github.com/commaai/openpilot/tree/master/tools)를 확인하고 그에 따라 필요 프로그램을 설치하세요.
- 여러분의 ~/.bashrc 에 "export PYTHONPATH=$HOME/openpilot" 라인을 추가하세요.
- tensorflow 2.2 버전과 nvidia 드라이버(nvidia-xxx/cuda10.0/cudnn7.6.5)를 설치하세요.
- [OpenCL 드라이버](http://registrationcenter-download.intel.com/akdlm/irc_nas/vcp/15532/l_opencl_p_18.1.0.015.tgz)를 설치하세요.
- [OpenCV4](https://www.pyimagesearch.com/2018/08/15/how-to-install-opencv-4-on-ubuntu/)를 설치하세요. (Python 파트는 무시하세요)

## 웹캠용 openpilot 빌드
```
cd ~/openpilot
```
- 거꾸로 된 카메라가 있는 경우 빌드하기 전에 selfdrive/camerad/cameras/camera_webcam.cc 라인 72 및 146을 확인하세요.
```
USE_WEBCAM=1 scons -j$(nproc)
```

## 하드웨어 연결
- 도로를 향하는 카메라를 먼저 연결한 다음 운전자를 향하는 카메라를 연결합니다.
- (기본 인덱스는 1, 2이며 selfdrive/camerad/cameras/camera_webcam.cc에서 수정할 수 있음)
- 컴퓨터를 panda에 연결

## 작동
```
cd ~/openpilot/selfdrive/manager
PASSIVE=0 NOSENSOR=1 USE_WEBCAM=1 ./manager.py
```
- 차에 시동을 걸면 UI에 도로 웹캠의 뷰가 표시됩니다.
- 웹캠 조정 및 보안(tools/webcam/front_mount_helper.py를 실행하여 드라이버 카메라 장착에 도움을 줄 수 있음)
- 보정을 완료하고 참여하세요!
