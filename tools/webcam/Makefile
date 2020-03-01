CXXFLAGS=-Wall -Wextra --std=c++11 -I$(HOME)/openpilot -I$(HOME)/openpilot/cereal/messaging -L$(HOME)/openpilot/cereal -lmessaging_shared `pkg-config --libs --cflags kj capnp opencv`
CXX=g++

SOURCES=webcam.cc

all:
	$(CXX) -o webcamerad $(SOURCES) $(CXXFLAGS)
