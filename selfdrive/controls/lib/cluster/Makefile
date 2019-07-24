CC = clang
CXX = clang++

CXXFLAGS = -Wall -g -fPIC -std=c++11 -O2

ifeq ($(ARCH),aarch64)
CXXFLAGS += -mcpu=cortex-a57
endif

OBJS = fastcluster.o test.o
DEPS := $(OBJS:.o=.d)

all: libfastcluster.so

test: libfastcluster.so test.o
	$(CXX) -g -L. -lfastcluster -o $@ $+

valgrind: test
	valgrind --leak-check=full ./test


libfastcluster.so: fastcluster.o
	$(CXX) -g -shared -o $@ $+

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -MMD -c $*.cpp

clean:
	rm -f $(OBJS) $(DEPS) libfastcluster.so test


-include $(DEPS)
