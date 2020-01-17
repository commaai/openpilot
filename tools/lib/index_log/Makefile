CC := gcc
CXX := g++
PHONELIBS?=../../../../phonelibs

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	CAPNP_FLAGS := /usr/local/lib/libcapnp.a /usr/local/lib/libkj.a
else
	CAPNP_FLAGS := -I $(PHONELIBS)/capnp-cpp/include -I $(PHONELIBS)/capnp-cpp/include
	CAPNP_LIBS := -L $(PHONELIBS)/capnp-cpp/x64/lib -L $(PHONELIBS)/capnp-cpp/x64/lib -l:libcapnp.a -l:libkj.a
endif

index_log: index_log.cc
	$(eval $@_TMP := $(shell mktemp))
	$(CXX) -std=gnu++11 -o $($@_TMP) \
    index_log.cc \
    $(CAPNP_FLAGS) \
    $(CAPNP_LIBS)
	mv $($@_TMP) $@
