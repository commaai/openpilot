CC = clang
CXX = clang++

ARCH := $(shell uname -m)
OS := $(shell uname -o)

BASEDIR = ../..
PHONELIBS = ../../phonelibs

WARN_FLAGS = -Werror=implicit-function-declaration \
             -Werror=incompatible-pointer-types \
             -Werror=int-conversion \
             -Werror=return-type \
             -Werror=format-extra-args

CFLAGS = -std=gnu11 -g -fPIC -I../ -I../../ -O2 $(WARN_FLAGS)
CXXFLAGS = -std=c++11 -g -fPIC -I../ -I../../ -O2 $(WARN_FLAGS)

ZMQ_LIBS = -l:libczmq.a -l:libzmq.a -lgnustl_shared

JSON_FLAGS = -I$(PHONELIBS)/json/src

EXTRA_LIBS = -lusb-1.0

# ifeq ($(OS),GNU/Linux)
# # for Drive PX2
# ZMQ_LIBS = -lczmq -lzmq
# CEREAL_LIBS = -lcapnp -lkj -lcapnp_c
# EXTRA_LIBS = -lusb-1.0 -lpthread
# endif
ifeq ($(ARCH),aarch64)
CFLAGS += -mcpu=cortex-a57
CXXFLAGS += -mcpu=cortex-a57
endif


ifeq ($(ARCH),x86_64)
ZMQ_FLAGS = -I$(PHONELIBS)/zmq/aarch64/include
ZMQ_LIBS = -L$(BASEDIR)/external/zmq/lib/ \
           -l:libczmq.a -l:libzmq.a
EXTRA_LIBS = -lusb-1.0 -lpthread
CXXFLAGS += -I/usr/include/libusb-1.0
CFLAGS += -I/usr/include/libusb-1.0
endif

.PHONY: all
all: boardd

include ../common/cereal.mk

OBJS = boardd.o \
       can_list_to_can_capnp.o \
       ../common/swaglog.o \
       ../common/params.o \
       ../common/util.o \
       $(PHONELIBS)/json/src/json.o \
       $(CEREAL_OBJS)

DEPS := $(OBJS:.o=.d)

boardd: $(OBJS)
	@echo "[ LINK ] $@"
	$(CXX) -fPIC -o '$@' $^ \
            $(CEREAL_LIBS) \
            $(ZMQ_LIBS) \
            $(EXTRA_LIBS)

boardd.o: boardd.cc
	@echo "[ CXX ] $@"
	$(CXX) $(CXXFLAGS) -MMD \
           -I$(PHONELIBS)/android_system_core/include \
           $(CEREAL_CFLAGS) \
           $(CEREAL_CXXFLAGS) \
           $(ZMQ_FLAGS) \
           -I../ \
           -I../../ \
           -c -o '$@' '$<'


boardd_api_impl.so: libcan_list_to_can_capnp.a boardd_api_impl.pyx boardd_setup.py
	python3 boardd_setup.py build_ext --inplace
	rm -rf build
	rm -f boardd_api_impl.cpp

libcan_list_to_can_capnp.a: can_list_to_can_capnp.o $(CEREAL_OBJS)
	ar rcsD '$@' $^

%.o: %.c
	@echo "[ CC ] $@"
	$(CC) $(CFLAGS) -MMD \
          -Iinclude -I.. -I../.. \
          $(CEREAL_CFLAGS) \
          $(ZMQ_FLAGS) \
          $(JSON_FLAGS) \
          -c -o '$@' '$<'

%.o: %.cc
	@echo "[ CC ] $@"
	$(CXX) $(CXXFLAGS) -MMD \
          -Iinclude -I.. -I../.. \
          $(CEREAL_CXXFLAGS) \
          $(ZMQ_FLAGS) \
          -c -o '$@' '$<'

.PHONY: clean
clean:
	rm -f boardd libcan_list_to_can_capnp.a boardd_api_impl.so $(OBJS) $(DEPS)

-include $(DEPS)
