CC = clang
CXX = clang++

PHONELIBS = ../../../phonelibs

WARN_FLAGS = -Werror=implicit-function-declaration \
             -Werror=incompatible-pointer-types \
             -Werror=int-conversion \
             -Werror=return-type \
             -Werror=format-extra-args \
             -Wno-deprecated-declarations

CFLAGS = -std=gnu11 -g -fPIC -O2 $(WARN_FLAGS)
CXXFLAGS = -std=c++1z -g -fPIC -O2 $(WARN_FLAGS)

FFMPEG_LIBS = -lavformat \
              -lavcodec \
              -lswscale \
              -lavutil \
              -lz \
              -lzmq 

OBJS = testraw.o \
       ../raw_logger.o \
       ../../common/visionipc.o \
       ../../common/ipc.o \
       ../../common/swaglog.o\
       $(PHONELIBS)/json11/json11.o 

testraw: $(OBJS)
	$(CXX) -fPIC -o '$@' $^ -L/usr/lib $(FFMPEG_LIBS)

%.o: %.cc
	@echo "[ CXX ] $@"
	$(CXX) $(CXXFLAGS) \
         -I../ \
         -I../../ \
         -I../../../ \
         -c -o '$@' '$<'
