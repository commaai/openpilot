CC = clang
CXX = clang++

ARCH := $(shell uname -m)
OS := $(shell uname -o)

PHONELIBS = ../../phonelibs

WARN_FLAGS = -Werror=implicit-function-declaration \
             -Werror=incompatible-pointer-types \
             -Werror=int-conversion \
             -Werror=return-type \
             -Werror=format-extra-args

CFLAGS = -std=gnu11 -g -fPIC -O2 $(WARN_FLAGS)
CXXFLAGS = -std=c++11 -g -fPIC -O2 $(WARN_FLAGS)

ZMQ_FLAGS = -I$(PHONELIBS)/zmq/aarch64/include
ZMQ_LIBS = -L$(PHONELIBS)/zmq/aarch64/lib \
           -l:libczmq.a -l:libzmq.a \
           -lgnustl_shared

JSON_FLAGS = -I$(PHONELIBS)/json/src

EXTRA_LIBS = -lusb

ifeq ($(OS),GNU/Linux)
# for Drive PX2
ZMQ_LIBS = -lczmq -lzmq
CEREAL_LIBS = -lcapnp -lkj -lcapnp_c
EXTRA_LIBS = -lusb-1.0 -lpthread
endif

ifeq ($(ARCH),x86_64)
ZMQ_LIBS = -L$(HOME)/one/external/zmq/lib/ \
           -l:libczmq.a -l:libzmq.a
CEREAL_LIBS = -L$(HOME)/one/external/capnp/lib/ \
              -l:libcapnp.a -l:libcapnp_c.a -l:libkj.a
EXTRA_LIBS = -lusb-1.0 -lpthread
endif

.PHONY: all
all: boardd

include ../common/cereal.mk

OBJS = boardd.o \
       ../common/swaglog.o \
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
	$(CXX) $(CXXFLAGS) \
           -I$(PHONELIBS)/android_system_core/include \
           $(CEREAL_CFLAGS) \
           $(CEREAL_CXXFLAGS) \
           $(ZMQ_FLAGS) \
           -I../ \
           -I../../ \
           -c -o '$@' '$<'

%.o: %.c
	@echo "[ CC ] $@"
	$(CC) $(CFLAGS) -MMD \
          -Iinclude -I.. -I../.. \
          $(CEREAL_CFLAGS) \
          $(ZMQ_FLAGS) \
          $(JSON_FLAGS) \
          -c -o '$@' '$<'

.PHONY: clean
clean:
	rm -f boardd $(OBJS) $(DEPS)

-include $(DEPS)
