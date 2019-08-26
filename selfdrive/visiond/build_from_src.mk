CC = clang
CXX = clang++

BASEDIR = ../..
EXTERNAL = ../../external
PHONELIBS = ../../phonelibs

WARN_FLAGS = -Werror=implicit-function-declaration \
             -Werror=incompatible-pointer-types \
             -Werror=int-conversion \
             -Werror=return-type \
             -Werror=format-extra-args \
             -Wno-deprecated-declarations

CFLAGS = -I. -std=gnu11 -fPIC -O2 $(WARN_FLAGS)
CXXFLAGS = -I. -std=c++14 -fPIC -O2 $(WARN_FLAGS)

ifeq ($(ARCH),aarch64)
CFLAGS += -mcpu=cortex-a57
CXXFLAGS += -mcpu=cortex-a57
endif

JSON_FLAGS = -I$(PHONELIBS)/json/src
JSON11_FLAGS = -I$(PHONELIBS)/json11/
EIGEN_FLAGS = -I$(PHONELIBS)/eigen

UNAME_M := $(shell uname -m)
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_M),x86_64)

ifeq ($(UNAME_S),Darwin)
  LIBYUV_FLAGS = -I$(PHONELIBS)/libyuv/include
  LIBYUV_LIBS = $(PHONELIBS)/libyuv/mac/lib/libyuv.a

  ZMQ_FLAGS = -I$(EXTERNAL)/zmq/include
  ZMQ_LIBS = $(PHONELIBS)/zmq/mac/lib/libczmq.a \
             $(PHONELIBS)/zmq/mac/lib/libzmq.a

  OPENCL_LIBS = -framework OpenCL

  PLATFORM_OBJS = cameras/camera_fake.o \
                  ../common/visionbuf_cl.o
else
  # assume x86_64 linux
  LIBYUV_FLAGS = -I$(PHONELIBS)/libyuv/include
  LIBYUV_LIBS = $(PHONELIBS)/libyuv/x64/lib/libyuv.a

  ZMQ_FLAGS = -I$(PHONELIBS)/zmq/x64/include
  ZMQ_LIBS = -L$(PHONELIBS)/zmq/x64/lib/ -l:libczmq.a -l:libzmq.a

  OPENCL_LIBS = -lOpenCL

  TF_FLAGS = -I$(EXTERNAL)/tensorflow/include
  TF_LIBS = -L$(EXTERNAL)/tensorflow/lib -ltensorflow \
            -Wl,-rpath $(EXTERNAL)/tensorflow/lib

  SNPE_FLAGS = -I$(PHONELIBS)/snpe/include/
  SNPE_LIBS = -L$(PHONELIBS)/snpe/x86_64-linux-clang/ \
              -lSNPE -lsymphony-cpu \
              -Wl,-rpath $(PHONELIBS)/snpe/x86_64-linux-clang/

  CFLAGS += -g
  CXXFLAGS += -g -I../common

  PLATFORM_OBJS = cameras/camera_frame_stream.o \
                  ../common/visionbuf_cl.o \
                  ../common/visionimg.o \
                  runners/tfmodel.o
endif

  SSL_FLAGS = -I/usr/include/openssl/
  SSL_LIBS = -lssl -lcrypto

  OTHER_LIBS = -lz -lm -lpthread

  CFLAGS += -D_GNU_SOURCE \
            -DCLU_NO_CACHE
  OBJS = visiond.o
else
	# assume phone

  LIBYUV_FLAGS = -I$(PHONELIBS)/libyuv/include
  LIBYUV_LIBS = $(PHONELIBS)/libyuv/lib/libyuv.a

  ZMQ_LIBS = -l:libczmq.a -l:libzmq.a -lgnustl_shared

  CURL_FLAGS = -I$(PHONELIBS)/curl/include
  CURL_LIBS = $(PHONELIBS)/curl/lib/libcurl.a \
              $(PHONELIBS)/zlib/lib/libz.a

  SSL_FLAGS = -I$(PHONELIBS)/boringssl/include
  SSL_LIBS = $(PHONELIBS)/boringssl/lib/libssl_static.a \
             $(PHONELIBS)/boringssl/lib/libcrypto_static.a

  OPENCL_FLAGS = -I$(PHONELIBS)/opencl/include
  OPENCL_LIBS = -lgsl -lCB -lOpenCL

  OPENGL_LIBS = -lGLESv3 -lEGL
  UUID_LIBS = -luuid

  SNPE_FLAGS = -I$(PHONELIBS)/snpe/include/
  SNPE_LIBS = -lSNPE -lsymphony-cpu -lsymphonypower

  OTHER_LIBS = -lz -lcutils -lm -llog -lui -ladreno_utils

  PLATFORM_OBJS = cameras/camera_qcom.o \
                  ../common/visionbuf_ion.o \
                  ../common/visionimg.o

  CFLAGS += -DQCOM \
	           -I$(PHONELIBS)/android_system_core/include \
						 -I$(PHONELIBS)/android_frameworks_native/include \
						 -I$(PHONELIBS)/android_hardware_libhardware/include \
	           -I$(PHONELIBS)/linux/include
  CXXFLAGS += -DQCOM \
	           -I$(PHONELIBS)/android_system_core/include \
						 -I$(PHONELIBS)/android_frameworks_native/include \
						 -I$(PHONELIBS)/android_hardware_libhardware/include \
	           -I$(PHONELIBS)/linux/include
  OBJS = visiond.o
endif

OUTPUT = visiond

.PHONY: all
all: $(OUTPUT)

include ../common/cereal.mk

OBJS += $(PLATFORM_OBJS) \
        ../common/swaglog.o \
        ../common/ipc.o \
        ../common/visionipc.o \
        ../common/util.o \
        ../common/params.o \
        ../common/efd.o \
        ../common/buffering.o \
        transforms/transform.o \
        transforms/loadyuv.o \
        transforms/rgb_to_yuv.o \
        models/commonmodel.o \
        runners/snpemodel.o \
        models/posenet.o \
        models/monitoring.o \
        models/driving.o \
        clutil.o \
        $(PHONELIBS)/json/src/json.o \
        $(PHONELIBS)/json11/json11.o \
        $(CEREAL_OBJS)

DEPS := $(OBJS:.o=.d)

rgb_to_yuv_test: transforms/rgb_to_yuv_test.o clutil.o transforms/rgb_to_yuv.o ../common/util.o
	@echo "[ LINK ] $@"
	$(CXX) -fPIC -o '$@' $^ \
        $(LIBYUV_LIBS) \
        $(LDFLAGS) \
        -L/usr/lib \
        -L/system/vendor/lib64 \
        $(OPENCL_LIBS) \


$(OUTPUT): $(OBJS)
	@echo "[ LINK ] $@"
	$(CXX) -fPIC -o '$@' $^ \
        $(LDFLAGS) \
        $(LIBYUV_LIBS) \
        $(OPENGL_LIBS) \
        $(CEREAL_LIBS) \
        $(ZMQ_LIBS) \
        -ljpeg \
        -L/usr/lib \
        -L/system/vendor/lib64 \
        $(OPENCL_LIBS) \
        $(CURL_LIBS) \
        $(SSL_LIBS) \
        $(TF_LIBS) \
        $(SNPE_LIBS) \
				$(UUID_LIBS) \
        $(OTHER_LIBS)

$(MODEL_OBJS): %.o: %.dlc
	@echo "[ bin2o ] $@"
	cd '$(dir $<)' && ld -r -b binary '$(notdir $<)' -o '$(abspath $@)'

%.o: %.cc
	@echo "[ CXX ] $@"
	$(CXX) $(CXXFLAGS) -MMD \
           -Iinclude -I.. -I../.. \
           $(EIGEN_FLAGS) \
           $(ZMQ_FLAGS) \
           $(CEREAL_CXXFLAGS) \
           $(OPENCL_FLAGS) \
           $(LIBYUV_FLAGS) \
           $(TF_FLAGS) \
           $(SNPE_FLAGS) \
           $(JSON_FLAGS) \
           $(JSON11_FLAGS) $(CURL_FLAGS) \
           -I$(PHONELIBS)/libgralloc/include \
           -I$(PHONELIBS)/linux/include \
           -c -o '$@' '$<'

%.o: %.c
	@echo "[ CC ] $@"
	$(CC) $(CFLAGS) -MMD \
          -Iinclude -I.. -I../.. \
          $(ZMQ_FLAGS) \
          $(CEREAL_CFLAGS) \
          $(OPENCL_FLAGS) \
          $(LIBYUV_FLAGS) \
          $(JSON_FLAGS) \
          -I$(PHONELIBS)/libgralloc/include \
          -I$(PHONELIBS)/linux/include \
          -c -o '$@' '$<'

.PHONY: clean
clean:
	rm -f visiond rgb_to_yuv_test rgb_to_yuv_test.o $(OBJS) $(DEPS)

-include $(DEPS)
