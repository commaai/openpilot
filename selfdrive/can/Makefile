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
LDFLAGS =

ifeq ($(ARCH),aarch64)
CFLAGS += -mcpu=cortex-a57
CXXFLAGS += -mcpu=cortex-a57
endif

ifeq ($(UNAME_S),Darwin)
	ZMQ_LIBS = -L/usr/local/lib -lzmq
else ifeq ($(OPTEST),1)
	ZMQ_LIBS = -lzmq
else ifeq ($(UNAME_M),x86_64)
	ZMQ_FLAGS = -I$(PHONELIBS)/zmq/x64/include
	ZMQ_LIBS = -L$(PHONELIBS)/zmq/x64/lib -l:libzmq.a
else ifeq ($(UNAME_M),aarch64)
	ZMQ_LIBS = -l:libzmq.a -lgnustl_shared
endif

OBJDIR = obj

OPENDBC_PATH := $(shell python2 -c 'import opendbc; print opendbc.DBC_PATH')

DBC_SOURCES := $(wildcard $(OPENDBC_PATH)/*.dbc)
DBC_OBJS := $(patsubst $(OPENDBC_PATH)/%.dbc,$(OBJDIR)/%.o,$(DBC_SOURCES))
DBC_CCS := $(patsubst $(OPENDBC_PATH)/%.dbc,dbc_out/%.cc,$(DBC_SOURCES))
.SECONDARY: $(DBC_CCS)

LIBDBC_OBJS := $(OBJDIR)/dbc.o $(OBJDIR)/parser.o $(OBJDIR)/packer.o

CWD := $(shell pwd)

.PHONY: all
all: $(OBJDIR) libdbc.so parser_pyx.so

include ../common/cereal.mk

# make sure cereal is built
libdbc.so:: ../../cereal/gen/cpp/log.capnp.h

../../cereal/gen/cpp/log.capnp.h:
	cd ../../cereal && make

libdbc.so:: $(LIBDBC_OBJS) $(DBC_OBJS)
	@echo "[ LINK ] $@"
	$(CXX) -fPIC -shared -o '$@' $^ \
		-I. -I../.. \
		$(CXXFLAGS) \
		$(LDFLAGS) \
		$(ZMQ_FLAGS) \
		$(ZMQ_LIBS) \
		$(CEREAL_CXXFLAGS) \
		$(CEREAL_LIBS)

packer_impl.so: packer_impl.pyx packer_setup.py
	python2 packer_setup.py build_ext --inplace
	rm -rf build
	rm -f packer_impl.cpp

parser_pyx.so: parser_pyx_setup.py parser_pyx.pyx parser_pyx.pxd
	python $< build_ext --inplace
	rm -rf build
	rm -f parser_pyx.cpp

$(OBJDIR)/%.o: %.cc
	@echo "[ CXX ] $@"
	$(CXX) -fPIC -c -o '$@' $^ \
		-I. -I../.. \
		$(CXXFLAGS) \
		$(ZMQ_FLAGS) \
		$(CEREAL_CXXFLAGS) \

$(OBJDIR)/%.o: dbc_out/%.cc
	@echo "[ CXX ] $@"
	$(CXX) -fPIC -c -o '$@' $^ \
		-I. -I../.. \
		$(CXXFLAGS) \
		$(ZMQ_FLAGS) \
		$(CEREAL_CXXFLAGS) \

dbc_out/%.cc: process_dbc.py dbc_template.cc $(OPENDBC_PATH)/%.dbc
	@echo "[ DBC GEN ] $@"
	@echo "Missing prereq $?"
	./process_dbc.py $(OPENDBC_PATH) dbc_out

$(OBJDIR):
	mkdir -p $@

.PHONY: clean $(OBJDIR)
clean:
	rm -rf libdbc.so*
	rm -f dbc_out/*.cc
	rm -f dbcs.txt
	rm -f dbcs.csv
	rm -rf $(OBJDIR)/*
