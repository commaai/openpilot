CC = clang
CXX = clang++


PHONELIBS = ../../phonelibs

WARN_FLAGS = -Werror=implicit-function-declaration \
             -Werror=incompatible-pointer-types \
             -Werror=int-conversion \
             -Werror=return-type \
             -Werror=format-extra-args

CFLAGS = -std=gnu11 -fPIC -O2 $(WARN_FLAGS)
CXXFLAGS = -std=c++11 -fPIC -O2 $(WARN_FLAGS)

ZMQ_FLAGS = -I$(PHONELIBS)/zmq/aarch64/include
ZMQ_LIBS = -L$(PHONELIBS)/zmq/aarch64/lib \
           -l:libczmq.a -l:libzmq.a \
           -lgnustl_shared

CEREAL_CFLAGS = -I$(PHONELIBS)/capnp-c/include
CEREAL_LIBS = -L$(PHONELIBS)/capnp-c/aarch64/lib -l:libcapn.a
CEREAL_OBJS = ../../cereal/gen/c/log.capnp.o ../../cereal/gen/c/car.capnp.o

NANOVG_FLAGS = -I$(PHONELIBS)/nanovg
JSON_FLAGS = -I$(PHONELIBS)/json/src

OPENCL_FLAGS = -I$(PHONELIBS)/opencl/include
OPENCL_LIBS = -lgsl -lCB -lOpenCL

OPENGL_LIBS = -lGLESv3

OPENSL_LIBS = -lOpenSLES

FRAMEBUFFER_LIBS = -lutils -lgui -lEGL

CFLAGS += -DQCOM
CXXFLAGS += -DQCOM

OBJS = slplay.o \
       ui.o \
       ../common/glutil.o \
       ../common/visionipc.o \
       ../common/ipc.o \
       ../common/visionimg.o \
       ../common/visionbuf_ion.o \
       ../common/framebuffer.o \
       ../common/params.o \
       ../common/util.o \
       ../common/touch.o \
       ../common/swaglog.o \
       $(PHONELIBS)/nanovg/nanovg.o \
       $(PHONELIBS)/json/src/json.o \
       $(CEREAL_OBJS)

DEPS := $(OBJS:.o=.d)

all: ui

ui: $(OBJS)
	@echo "[ LINK ] $@"
	$(CXX) -fPIC -o '$@' $^ \
	      $(FRAMEBUFFER_LIBS) \
	      $(CEREAL_LIBS) \
	      $(ZMQ_LIBS) \
        -L/system/vendor/lib64 \
        -lhardware -lui \
        $(OPENGL_LIBS) \
        $(OPENCL_LIBS) \
        ${OPENSL_LIBS} \
        -Wl,-rpath=/system/lib64,-rpath=/system/comma/usr/lib \
        -lcutils -lm -llog -lui -ladreno_utils

slplay.o: slplay.c
	@echo "[ CC ] $@"
	$(CC) $(CFLAGS) -fPIC \
	-I../ \
	$(OPENSL_LIBS) \
	-c -o '$@' $^

%.o: %.cc
	@echo "[ CXX ] $@"
	$(CXX) $(CXXFLAGS) -MMD \
           -Iinclude -I.. -I../.. \
           $(OPENCL_FLAGS) \
           -I$(PHONELIBS)/android_frameworks_native/include \
           -I$(PHONELIBS)/android_system_core/include \
           -I$(PHONELIBS)/android_hardware_libhardware/include \
           -I$(PHONELIBS)/libgralloc/include \
           -I$(PHONELIBS)/linux/include \
           -c -o '$@' '$<'

%.o: %.c
	@echo "[ CC ] $@"
	$(CC) $(CFLAGS) -MMD \
          -Iinclude -I.. -I../.. \
          $(NANOVG_FLAGS) \
          $(ZMQ_FLAGS) \
          $(CEREAL_CFLAGS) \
          $(JSON_FLAGS) \
          $(OPENCL_FLAGS) \
          -I$(PHONELIBS)/linux/include \
          -c -o '$@' '$<'

.PHONY: clean
clean:
	rm -f ui $(OBJS) $(DEPS)

-include $(DEPS)
