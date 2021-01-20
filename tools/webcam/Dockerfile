FROM commaai/openpilot-base:latest

ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /tmp/openpilot:${PYTHONPATH}

# install opencv
ENV OPENCV_VERSION '4.2.0'
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      libvtk6-dev \
      libdc1394-22-dev \
      libavcodec-dev \
      libavformat-dev \
      libswscale-dev \
      libtheora-dev \
      libvorbis-dev \
      libxvidcore-dev \
      libx264-dev \
      yasm \
      libopencore-amrnb-dev \
      libopencore-amrwb-dev \
      libv4l-dev \
      libxine2-dev \
      libtbb-dev \
    && rm -rf /var/lib/apt/lists/* && \

    mkdir /tmp/opencv_build && \
    cd /tmp/opencv_build && \

    wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz && \
    tar -xvf ${OPENCV_VERSION}.tar.gz && \
    mv opencv-${OPENCV_VERSION} OpenCV && \
    cd OpenCV && mkdir build && cd build && \
    cmake -DWITH_OPENGL=ON -DFORCE_VTK=ON -DWITH_TBB=ON -DWITH_GDAL=ON \
          -DWITH_XINE=ON -DENABLE_PRECOMPILED_HEADERS=OFF .. && \
    make -j8 && \
    make install && \
    ldconfig && \

    cd /tmp && rm -rf /tmp/opencv_build


RUN mkdir -p /tmp/openpilot

COPY SConstruct \
     .pylintrc \
     .pre-commit-config.yaml \
     /tmp/openpilot/

COPY ./pyextra /tmp/openpilot/pyextra
COPY ./phonelibs /tmp/openpilot/phonelibs
COPY ./site_scons /tmp/openpilot/site_scons
COPY ./laika /tmp/openpilot/laika
COPY ./laika_repo /tmp/openpilot/laika_repo
COPY ./rednose /tmp/openpilot/rednose
COPY ./tools /tmp/openpilot/tools
COPY ./release /tmp/openpilot/release
COPY ./common /tmp/openpilot/common
COPY ./opendbc /tmp/openpilot/opendbc
COPY ./cereal /tmp/openpilot/cereal
COPY ./panda /tmp/openpilot/panda
COPY ./selfdrive /tmp/openpilot/selfdrive
