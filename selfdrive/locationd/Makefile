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

CFLAGS = -std=gnu11 -g -fPIC -I../ -I../../ -O2 $(WARN_FLAGS) -Wall
CXXFLAGS = -std=c++11 -g -fPIC -I../ -I../../ -O2 $(WARN_FLAGS) -Wall
ZMQ_LIBS = -l:libczmq.a -l:libzmq.a

ifeq ($(ARCH),aarch64)
CFLAGS += -mcpu=cortex-a57
CXXFLAGS += -mcpu=cortex-a57
ZMQ_LIBS += -lgnustl_shared
endif


JSON_FLAGS = -I$(PHONELIBS)/json/src
JSON11_FLAGS = -I$(PHONELIBS)/json11

EXTRA_LIBS = -lpthread

ifeq ($(ARCH),x86_64)
ZMQ_FLAGS = -I$(BASEDIR)/phonelibs/zmq/x64/include
ZMQ_LIBS = -L$(BASEDIR)/external/zmq/lib \
           -l:libczmq.a -l:libzmq.a
ZMQ_SHARED_LIBS = -L$(BASEDIR)/external/zmq/lib \
                  -lczmq -lzmq
else
EXTRA_LIBS += -llog -luuid
endif

.PHONY: all
all: ubloxd params_learner

include ../common/cereal.mk

LOC_OBJS = locationd_yawrate.o params_learner.o \
           ../common/swaglog.o \
           ../common/params.o \
           ../common/util.o \
					 $(PHONELIBS)/json11/json11.o \
					 $(PHONELIBS)/json/src/json.o \
           $(CEREAL_OBJS)

LOC_DEPS := $(LOC_OBJS:.o=.d)

OBJS = ublox_msg.o \
       ubloxd_main.o \
       ../common/swaglog.o \
       ../common/params.o \
       ../common/util.o \
       $(PHONELIBS)/json/src/json.o \
       $(CEREAL_OBJS)

DEPS := $(OBJS:.o=.d) ubloxd.d ubloxd_test.d

liblocationd.so: $(LOC_OBJS)
	@echo "[ LINK ] $@"
	$(CXX) -shared -o '$@' $^ \
            $(CEREAL_LIBS) \
            $(ZMQ_SHARED_LIBS) \
            $(EXTRA_LIBS)

params_learner: $(LOC_OBJS)
	@echo "[ LINK ] $@"
	$(CXX) -fPIC -o '$@' $^ \
            $(CEREAL_LIBS) \
            $(ZMQ_LIBS) \
            $(EXTRA_LIBS)

ubloxd: ubloxd.o $(OBJS)
	@echo "[ LINK ] $@"
	$(CXX) -fPIC -o '$@' $^ \
            $(CEREAL_LIBS) \
            $(ZMQ_LIBS) \
            $(EXTRA_LIBS)

ubloxd_test: ubloxd_test.o $(OBJS)
	@echo "[ LINK ] $@"
	$(CXX) -fPIC -o '$@' $^ \
            $(CEREAL_LIBS) \
            $(ZMQ_LIBS) \
            $(EXTRA_LIBS)

%.o: %.cc
	@echo "[ CXX ] $@"
	$(CXX) $(CXXFLAGS) -MMD \
          -Iinclude -I.. -I../.. \
           $(CEREAL_CXXFLAGS) \
           $(ZMQ_FLAGS) \
           $(JSON11_FLAGS) \
           $(JSON_FLAGS) \
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
	rm -f ubloxd params_learner liblocationd.so ubloxd.d ubloxd.o ubloxd_test ubloxd_test.o ubloxd_test.d $(OBJS) $(LOC_OBJS) $(DEPS)

-include $(DEPS)
-include $(LOC_DEPS)
