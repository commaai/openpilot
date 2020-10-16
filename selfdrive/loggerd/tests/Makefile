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

OBJS = test_encoder.o \
       ../encoder.o \
       ../raw_logger.o \
       ../../common/swaglog.o\
       ../../common/cqueue.o \
        ../../common/visionipc.o \
       ../../common/ipc.o \
       $(PHONELIBS)/json11/json11.o 

ENCODER_LIBS = -lOmxVenc \
       -lOmxCore \
       -llog \
       -lavformat \
       -lavcodec \
       -lswscale \
       -lavutil \
       -lzmq \
       -lyuv

%.o: %.cc
	@echo "[ CXX ] $@"
	$(CXX) $(CXXFLAGS) \
         -I../ \
         -I../../ \
         -I../../../ \
         -I$(PHONELIBS)/openmax/include \
         -I$(PHONELIBS)/libyuv/include \
         -I$(PHONELIBS)/json11 \
         -c -o '$@' '$<'


test_encoder: $(OBJS)
	$(CXX) -fPIC -o '$@' $^ -L$(PHONELIBS)/libyuv/lib -L $(ENCODER_LIBS) 