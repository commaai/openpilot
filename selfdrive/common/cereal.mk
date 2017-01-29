CEREAL_CFLAGS = -I$(PHONELIBS)/capnp-c/include
CEREAL_CXXFLAGS = -I$(PHONELIBS)/capnp-cpp/include
CEREAL_LIBS = -L$(PHONELIBS)/capnp-cpp/aarch64/lib/ \
              -L$(PHONELIBS)/capnp-c/aarch64/lib/ \
              -l:libcapn.a -l:libcapnp.a -l:libkj.a
CEREAL_OBJS = ../../cereal/gen/c/log.capnp.o ../../cereal/gen/c/car.capnp.o

log.capnp.o: ../../cereal/gen/cpp/log.capnp.c++
	@echo "[ CXX ] $@"
	$(CXX) $(CXXFLAGS) $(CEREAL_CFLAGS) \
           -c -o '$@' '$<'

car.capnp.o: ../../cereal/gen/cpp/car.capnp.c++
	@echo "[ CXX ] $@"
	$(CXX) $(CXXFLAGS) $(CEREAL_CFLAGS) \
           -c -o '$@' '$<'
  
