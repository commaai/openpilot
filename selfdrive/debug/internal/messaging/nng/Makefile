CC = clang
CXX = clang++

ARCH := $(shell uname -m)
OS := $(shell uname -o)

BASEDIR = ../../../..
PHONELIBS = ../../../../phonelibs

WARN_FLAGS = -Werror=implicit-function-declaration \
             -Werror=incompatible-pointer-types \
             -Werror=int-conversion \
             -Werror=return-type \
             -Werror=format-extra-args

CFLAGS = -std=gnu11 -g -fPIC -O2 $(WARN_FLAGS) -Wall
CXXFLAGS = -std=c++11 -g -fPIC -O2 $(WARN_FLAGS) -Wall
NNG_LIBS = -l:libnng.a

ifeq ($(ARCH),aarch64)
CFLAGS += -mcpu=cortex-a57
CXXFLAGS += -mcpu=cortex-a57
endif


EXTRA_LIBS = -lpthread

ifeq ($(ARCH),x86_64)
ZMQ_FLAGS = -I$(BASEDIR)/phonelibs/nng/x64/include
NNG_LIBS = -L$(BASEDIR)/phonelibs/nng/x64/lib \
           -l:libnng.a
else
EXTRA_LIBS += -llog -luuid
endif

.PHONY: all
all: sender receiver

receiver: receiver.o
	@echo "[ LINK ] $@"
	$(CXX) -fPIC -o '$@' $^ \
            $(CEREAL_LIBS) \
            $(NNG_LIBS) \
            $(EXTRA_LIBS)

sender: sender.o
	@echo "[ LINK ] $@"
	$(CXX) -fPIC -o '$@' $^ \
            $(CEREAL_LIBS) \
            $(NNG_LIBS) \
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


.PHONY: clean
clean:
	rm -f *.d sender receiver
