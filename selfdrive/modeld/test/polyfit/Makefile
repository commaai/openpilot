PHONELIBS = ../../../../phonelibs

EIGEN_FLAGS = -I$(PHONELIBS)/eigen

CXXFLAGS += $(EIGEN_FLAGS)
LDFLAGS += -lm

.PHONY: clean

main: main.cc data.h
	g++ -O2 $(EIGEN_FLAGS) -o main main.cc -lm


clean:
	rm -f main
