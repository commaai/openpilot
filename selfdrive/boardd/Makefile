CC = clang
CXX = clang++

ARCH := $(shell uname -m)

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

CEREAL_FLAGS = -I$(PHONELIBS)/capnp-cpp/include
CEREAL_LIBS = -L$(PHONELIBS)/capnp-cpp/aarch64/lib/ \
              -l:libcapnp.a -l:libkj.a
CEREAL_OBJS = ../../cereal/gen/c/log.capnp.o

EXTRA_LIBS = -lusb

ifeq ($(ARCH),x86_64)
ZMQ_LIBS = -L$(HOME)/drive/external/zmq/lib/ \
           -l:libczmq.a -l:libzmq.a
CEREAL_LIBS = -L$(HOME)/drive/external/capnp/lib/ \
              -l:libcapnp.a -l:libkj.a
EXTRA_LIBS = -lusb-1.0 -lpthread
endif


OBJS = boardd.o \
       log.capnp.o

DEPS := $(OBJS:.o=.d)

all: boardd

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
           $(CEREAL_FLAGS) \
           $(ZMQ_FLAGS) \
           -I../ \
           -I../../ \
           -c -o '$@' '$<'


log.capnp.o: ../../cereal/gen/cpp/log.capnp.c++
	@echo "[ CXX ] $@"
	$(CXX) $(CXXFLAGS) $(CEREAL_FLAGS) \
           -c -o '$@' '$<'
  

.PHONY: clean
clean:
	rm -f boardd $(OBJS) $(DEPS)

-include $(DEPS)
