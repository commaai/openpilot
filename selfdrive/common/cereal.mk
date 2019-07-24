UNAME_M ?= $(shell uname -m)
UNAME_S ?= $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
CEREAL_CFLAGS = -I$(PHONELIBS)/capnp-c/include
CEREAL_CXXFLAGS = -I$(PHONELIBS)/capnp-cpp/mac/include
CEREAL_LIBS = $(PHONELIBS)/capnp-cpp/mac/lib/libcapnp.a \
              $(PHONELIBS)/capnp-cpp/mac/lib/libkj.a \
              $(PHONELIBS)/capnp-c/mac/lib/libcapnp_c.a

else ifeq ($(UNAME_M),x86_64)
CEREAL_CFLAGS = -I$(PHONELIBS)/capnp-c/include
CEREAL_CXXFLAGS = -I$(PHONELIBS)/capnp-cpp/include
ifeq ($(CEREAL_LIBS),)
  CEREAL_LIBS = -L$(PHONELIBS)/capnp-cpp/x64/lib/ \
                -L$(PHONELIBS)/capnp-c/x64/lib/ \
                -l:libcapnp.a -l:libkj.a -l:libcapnp_c.a
endif

else

#CEREAL_CXXFLAGS = -I$(PHONELIBS)/capnp-cpp/include
ifeq ($(CEREAL_LIBS),)
  CEREAL_LIBS = -l:libcapn.a -l:libcapnp.a -l:libkj.a
endif
endif

CEREAL_OBJS = ../../cereal/gen/c/log.capnp.o ../../cereal/gen/c/car.capnp.o

log.capnp.o: ../../cereal/gen/cpp/log.capnp.c++
	@echo "[ CXX ] $@"
	$(CXX) $(CXXFLAGS) $(CEREAL_CXXFLAGS) \
           -c -o '$@' '$<'

car.capnp.o: ../../cereal/gen/cpp/car.capnp.c++
	@echo "[ CXX ] $@"
	$(CXX) $(CXXFLAGS) $(CEREAL_CXXFLAGS) \
           -c -o '$@' '$<'
