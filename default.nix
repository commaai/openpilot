with (import <nixpkgs> {});

mkShell {
  buildInputs = [
    python38
    pipenv
    capnproto
    scons
    clang
    opencl-headers
    ocl-icd
    darwin.apple_sdk.frameworks.OpenGL
    darwin.apple_sdk.frameworks.OpenCL
    zeromq
    eigen
    gcc-arm-embedded
    libusb
    cmake
    libffi
    ffmpeg
    libjpeg
    openssl
    bzip2
    curl
    # qt5.qtbase
  ];
}