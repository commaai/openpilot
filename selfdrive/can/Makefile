CC = clang
CXX = clang++

BASEDIR = ../..
PHONELIBS := ../../phonelibs

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

WARN_FLAGS = -Werror=implicit-function-declaration \
             -Werror=incompatible-pointer-types \
             -Werror=int-conversion \
             -Werror=return-type \
             -Werror=format-extra-args \
             -Wno-deprecated-declarations

CFLAGS = -std=gnu11 -g -fPIC -O2 $(WARN_FLAGS)
CXXFLAGS = -std=c++11 -g -fPIC -O2 $(WARN_FLAGS)

ifeq ($(UNAME_S),Darwin)
	CEREAL_LIBS := -L /usr/local/lib -lkj -lcapnp
	ZMQ_LIBS = -L/usr/local/lib -lzmq
else ifeq ($(OPTEST),1)
	ZMQ_LIBS = -lzmq
	CEREAL_LIBS = -lcapnp -lkj
else ifeq ($(UNAME_M),x86_64)
	EXTERNAL := ../../external
	ZMQ_FLAGS = -I$(EXTERNAL)/zmq/include
	ZMQ_LIBS = -L$(EXTERNAL)/zmq/lib -l:libzmq.a
	CEREAL_CXXFLAGS := -I$(EXTERNAL)/capnp/include
	CEREAL_LIBS := -L$(EXTERNAL)/capnp/lib -l:libcapnp.a -l:libkj.a
else ifeq ($(UNAME_M),aarch64)
	ZMQ_FLAGS = -I$(PHONELIBS)/zmq/aarch64/include
	ZMQ_LIBS = -L$(PHONELIBS)/zmq/aarch64/lib -l:libzmq.a
endif

OPENDBC_PATH := $(shell python -c 'import opendbc; print opendbc.DBC_PATH')

DBC_SOURCES := $(wildcard $(OPENDBC_PATH)/*.dbc)
DBC_CCS := $(patsubst $(OPENDBC_PATH)/%.dbc,dbc_out/%.cc,$(DBC_SOURCES))

CWD := $(shell pwd)

.PHONY: all
all: libdbc.so

ifeq ($(UNAME_M),aarch64)
include ../common/cereal.mk
endif

# make sure cereal is built
libdbc.so:: ../../cereal/gen/cpp/log.capnp.h

../../cereal/gen/cpp/log.capnp.h:
	cd ../../cereal && make

libdbc.so:: dbc.cc parser.cc packer.cc $(DBC_CCS)
	$(CXX) -fPIC -shared -o '$@' $^ \
	  -I. \
	  -I../.. \
    $(CXXFLAGS) \
    $(ZMQ_FLAGS) \
    $(ZMQ_LIBS) \
    $(CEREAL_CXXFLAGS) \
    $(CEREAL_LIBS)

dbc_out/%.cc: $(OPENDBC_PATH)/%.dbc process_dbc.py dbc_template.cc
	PYTHONPATH=$(PYTHONPATH):$(CWD)/../../pyextra ./process_dbc.py '$<' '$@'

.PHONY: clean
clean:
	rm -rf libdbc.so*
	rm -f dbc_out/*.cc
