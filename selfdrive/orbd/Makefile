# CPU 

CC = clang
CXX = clang++

WARN_FLAGS = -Werror=implicit-function-declaration \
             -Werror=incompatible-pointer-types \
             -Werror=int-conversion \
             -Werror=return-type \
             -Werror=format-extra-args

JSON_FLAGS = -I$(PHONELIBS)/json/src

CFLAGS = -std=gnu11 -g -O2 -fPIC $(WARN_FLAGS) -Iinclude $(JSON_FLAGS) -I.
CXXFLAGS = -std=c++11 -g -O2 -fPIC $(WARN_FLAGS) -Iinclude $(JSON_FLAGS) -I.
LDFLAGS =

# profile
# CXXFLAGS += -DTURBOCV_PROFILE=1

PHONELIBS = ../../phonelibs
BASEDIR = ../..
EXTERNAL = ../../external
PYTHONLIBS = 

UNAME_M := $(shell uname -m)

ifeq ($(UNAME_M),x86_64)
# computer

ZMQ_FLAGS = -I$(PHONELIBS)/zmq/aarch64/include
ZMQ_LIBS = -L$(BASEDIR)/external/zmq/lib/ \
           -l:libczmq.a -l:libzmq.a -lpthread

OPENCV_LIBS = -lopencv_core -lopencv_highgui -lopencv_features2d -lopencv_imgproc

CXXFLAGS += -fopenmp
LDFLAGS += -lomp

else
# phone
ZMQ_FLAGS = -I$(PHONELIBS)/zmq/aarch64/include
ZMQ_LIBS = -L$(PHONELIBS)/zmq/aarch64/lib \
	-l:libczmq.a -l:libzmq.a \
	-lgnustl_shared

OPENCV_FLAGS = -I$(PHONELIBS)/opencv/include
OPENCV_LIBS = -Wl,--enable-new-dtags -Wl,-rpath,/usr/local/lib/python2.7/site-packages -L/usr/local/lib/python2.7/site-packages -l:cv2.so

endif

.PHONY: all
all: orbd

include ../common/cereal.mk

DEP_OBJS = ../common/visionipc.o ../common/ipc.o ../common/swaglog.o $(PHONELIBS)/json/src/json.o

orbd: orbd_dsp.o $(DEP_OBJS) calculator_stub.o freethedsp.o
	@echo "[ LINK ] $@"
	$(CXX) -fPIC -o '$@' $^ \
    $(LDFLAGS) \
		$(ZMQ_LIBS) \
		$(CEREAL_LIBS) \
		-L/usr/lib \
		-L/system/vendor/lib64 \
		-ladsprpc \
		-lm -lz -llog

%.o: %.c
	@echo "[ CC ] $@"
	$(CC) $(CFLAGS) \
		$(ZMQ_FLAGS) \
		-I../ \
		-I../../ \
		-c -o '$@' '$<'

orbd_dsp.o: orbd.cc
	@echo "[ CXX ] $@"
	$(CXX) $(CXXFLAGS) \
		$(CEREAL_CXXFLAGS) \
		$(ZMQ_FLAGS) \
		$(OPENCV_FLAGS) \
		-DDSP \
		-I../ \
		-I../../ \
		-I../../../ \
		-I./include \
		-c -o '$@' '$<'

freethedsp.o: dsp/freethedsp.c
	@echo "[ CC ] $@"
	$(CC) $(CFLAGS) \
		-c -o '$@' '$<'

calculator_stub.o: dsp/gen/calculator_stub.c
	@echo "[ CC ] $@"
	$(CC) $(CFLAGS) -I./include -c -o '$@' '$<'

-include internal.mk

.PHONY: clean
clean:
	rm -f *.o turbocv.so orbd test/turbocv_profile test/turbocv_test test/*.o *_lut.h

