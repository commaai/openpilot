SRCS := log.capnp car.capnp

GENS := gen/cpp/car.capnp.c++ gen/cpp/log.capnp.c++


UNAME_M ?= $(shell uname -m)

# only generate C++ for docker tests
ifneq ($(OPTEST),1)
	GENS += gen/c/car.capnp.c gen/c/log.capnp.c gen/c/c++.capnp.h gen/c/java.capnp.h

# Dont build java on the phone...
ifeq ($(UNAME_M),x86_64)
	GENS += gen/java/Car.java gen/java/Log.java
endif

endif

.PHONY: all
all: $(GENS)

.PHONY: clean
clean:
	rm -rf gen

gen/c/%.capnp.c: %.capnp
	@echo "[ CAPNPC C ] $@"
	mkdir -p gen/c/
	capnpc '$<' -o c:gen/c/

gen/cpp/%.capnp.c++: %.capnp
	@echo "[ CAPNPC C++ ] $@"
	mkdir -p gen/cpp/
	capnpc '$<' -o c++:gen/cpp/

gen/java/Car.java gen/java/Log.java: $(SRCS)
	@echo "[ CAPNPC java ] $@"
	mkdir -p gen/java/
	capnpc $^ -o java:gen/java

# c-capnproto needs some empty headers
gen/c/c++.capnp.h gen/c/java.capnp.h:
	mkdir -p gen/c/
	touch '$@'

