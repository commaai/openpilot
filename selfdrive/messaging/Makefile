CXX := clang++
CC := clang

BASEDIR = ../..
PHONELIBS = ../../phonelibs

CXXFLAGS := -g -O3 -fPIC -std=c++11 -Wall -Wextra -Wshadow -Weffc++ -Wstrict-aliasing -Wpedantic -Werror -MMD -I$(BASEDIR)/selfdrive

LDLIBS=-lm -lstdc++ -lrt -lpthread

UNAME_M := $(shell uname -m)

YAML_FLAGS = -I$(PHONELIBS)/yaml-cpp/include
YAML_LIB = $(abspath $(PHONELIBS)/yaml-cpp/lib/libyaml-cpp.a)

ifeq ($(UNAME_M),aarch64)
	LDFLAGS += -llog -lgnustl_shared
	ZMQ_LIBS = /usr/lib/libzmq.a
endif
ifeq ($(UNAME_M),x86_64)
	ZMQ_FLAGS = -I$(BASEDIR)/phonelibs/zmq/x64/include
	ZMQ_LIBS = $(abspath $(PHONELIBS)/zmq/x64/lib/libzmq.a)
	YAML_DIR = $(PHONELIBS)/yaml-cpp/x64/lib/
	YAML_LIB = $(abspath $(PHONELIBS)/yaml-cpp/x64/lib/libyaml-cpp.a)
endif

ifdef ASAN
	CXXFLAGS += -fsanitize=address -fno-omit-frame-pointer
  LDFLAGS += -fsanitize=address
endif

CXXFLAGS += $(ZMQ_FLAGS) $(YAML_FLAGS)

OBJS := messaging.o impl_zmq.o
DEPS=$(OBJS:.o=.d)

.PRECIOUS: $(OBJS)
.PHONY: all clean
all: messaging.a messaging_pyx.so

demo: messaging.a demo.o
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -L. -l:messaging.a -o '$@'

messaging_pyx.so: messaging.a messaging_pyx_setup.py messaging_pyx.pyx messaging.pxd
	python3 messaging_pyx_setup.py build_ext --inplace
	rm -rf build
	rm -f messaging_pyx.cpp

%.a: $(OBJS)
	@echo "[ LINK ] $@"
	mkdir -p libs; \
		cd libs; \
		ar -x $(ZMQ_LIBS); \
		ar -x $(YAML_LIB);

	ar rcsD '$@' $^ libs/*.o
	rm -r libs

clean:
	@echo "[ CLEAN ]"
	rm -rf *.so *.a demo libs $(OBJS) $(DEPS)

-include $(DEPS)
