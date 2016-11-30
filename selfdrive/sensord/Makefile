CC = clang
CXX = clang++

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

OBJS = sensors.o \
       log.capnp.o

DEPS := $(OBJS:.o=.d)

all: sensord

sensord: $(OBJS)
	@echo "[ LINK ] $@"
	$(CXX) -fPIC -o '$@' $^ \
            $(CEREAL_LIBS) \
            $(ZMQ_LIBS) \
            -lhardware

sensors.o: sensors.cc
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
	rm -f sensord $(OBJS) $(DEPS)

-include $(DEPS)
