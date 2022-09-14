FROM ghcr.io/commaai/openpilot-base:latest

ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /tmp/openpilot:${PYTHONPATH}

# Install opencv
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

    curl -L -O https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz && \
    tar -xvf ${OPENCV_VERSION}.tar.gz && \
    mv opencv-${OPENCV_VERSION} OpenCV && \
    cd OpenCV && mkdir build && cd build && \
    cmake -DWITH_OPENGL=ON -DFORCE_VTK=ON -DWITH_TBB=ON -DWITH_GDAL=ON \
          -DWITH_XINE=ON -DENABLE_PRECOMPILED_HEADERS=OFF -DBUILD_TESTS=OFF \
          -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF .. && \
    make -j8 && \
    make install && \
    ldconfig && \

    cd / && rm -rf /tmp/*
