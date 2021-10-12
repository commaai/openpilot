CC:=clang
CXX:=clang++
OPT_FLAGS:=-O2 -g -ggdb3

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	SHARED_FLAGS=-Wl,--whole-archive $^ -Wl,--no-whole-archive
endif
ifeq ($(UNAME_S),Darwin)
	SHARED_FLAGS=-Wl,-force_load $^
endif

PHONELIBS := ../../third_party
BASEDIR := ../..

WARN_FLAGS = -Werror=implicit-function-declaration \
             -Werror=incompatible-pointer-types \
             -Werror=int-conversion \
             -Werror=return-type \
             -Werror=format-extra-args

CFLAGS = -std=gnu11 -g -fPIC $(OPT_FLAGS) $(WARN_FLAGS)
CXXFLAGS = -std=c++1z -fPIC $(OPT_FLAGS) $(WARN_FLAGS)

EIGEN_FLAGS = -I$(PHONELIBS)/eigen

CEREAL_LIBS = $(BASEDIR)/cereal/libmessaging.a

OPENCV_LIBS = -lopencv_video -lopencv_core -lopencv_imgproc

ifeq ($(UNAME_S),Darwin)
  VT_LDFLAGS += $(PHONELIBS)/capnp-c/mac/lib/libcapnp_c.a \
                 $(PHONELIBS)/zmq/mac/lib/libzmq.a \
                -framework OpenCL

  OPENCV_LIBS += -L/usr/local/opt/opencv@2/lib
  OPENCV_FLAGS += -I/usr/local/opt/opencv@2/include

else
  VT_LDFLAGS += $(CEREAL_LIBS) \
								-L/system/vendor/lib64 \
                -L$(BASEDIR)/external/zmq/lib/ \
								-l:libzmq.a \
                -lOpenCL
endif

.PHONY: all visiontest clean test
all: visiontest

libvisiontest_inputs := visiontest.c \
                        transforms/transform.cc \
                        transforms/loadyuv.cc \
                        ../common/clutil.cc \
                        $(BASEDIR)/selfdrive/common/util.c \
                        $(CEREAL_OBJS)

visiontest: libvisiontest.so
all-tests := $(addsuffix .test, $(basename $(wildcard test_*)))

%.o: %.cc
	@echo "[ CXX ] $@"
	$(CXX) $(CXXFLAGS) -MMD \
		-I. -I.. -I../.. \
		-Wall \
		-I$(BASEDIR)/ -I$(BASEDIR)/selfdrive -I$(BASEDIR)/selfdrive/common \
		$(EIGEN_FLAGS) \
    $(OPENCV_FLAGS) \
    $(CEREAL_CXXFLAGS) \
		-c -o '$@' '$<'

%.o: %.c
	@echo "[ CXX ] $@"
	$(CC) $(CFLAGS) -MMD \
		-I. -I.. -I../.. \
		-Wall \
		-I$(BASEDIR)/ -I$(BASEDIR)/selfdrive -I$(BASEDIR)/selfdrive/common \
    $(CEREAL_CFLAGS) \
		-c -o '$@' '$<'

libvisiontest.so: $(libvisiontest_inputs)
	$(eval $@_TMP := $(shell mktemp))
	$(CC) -std=gnu11 -shared -fPIC -O2 -g \
		-Werror=implicit-function-declaration -Werror=incompatible-pointer-types \
		-Werror=int-conversion -Wno-pointer-to-int-cast \
		-I. \
		$^ -o $($@_TMP) \
		-I$(PHONELIBS)/opencl/include \
		-I$(BASEDIR)/selfdrive/common \
		$(CEREAL_CXXFLAGS) \
		$(CEREAL_CFLAGS) \
		-I$(BASEDIR)/external/zmq/include \
		-I$(BASEDIR)/ -I$(BASEDIR)/selfdrive \
		-lstdc++ \
		$(VT_LDFLAGS) \
		-lm -lpthread
	mv $($@_TMP) $@

test : $(all-tests)

test_%.test : test_%
	@./'$<' || echo FAIL

clean:
	rm -rf *.o *.so *.a
