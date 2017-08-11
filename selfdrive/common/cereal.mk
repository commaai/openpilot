UNAME_M ?= $(shell uname -m)
ifeq ($(UNAME_M),x86_64)

CEREAL_CFLAGS = -I$(ONE)/external/capnp/include
CEREAL_CXXFLAGS = $(CEREAL_CFLAGS)
ifeq ($(CEREAL_LIBS),)
	CEREAL_LIBS = -L$(ONE)/external/capnp/lib \
	              -l:libcapnp.a -l:libkj.a -l:libcapnp_c.a
endif

else
CEREAL_CFLAGS = -I$(PHONELIBS)/capnp-c/include
CEREAL_CXXFLAGS = -I$(PHONELIBS)/capnp-cpp/include
ifeq ($(CEREAL_LIBS),)
  CEREAL_LIBS = -L$(PHONELIBS)/capnp-cpp/aarch64/lib/ \
                -L$(PHONELIBS)/capnp-c/aarch64/lib/ \
                -l:libcapn.a -l:libcapnp.a -l:libkj.a
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
  
