CC = clang
CXX = clang++

PHONELIBS = ../../../../phonelibs

WARN_FLAGS = -Werror=implicit-function-declaration \
             -Werror=incompatible-pointer-types \
             -Werror=int-conversion \
             -Werror=return-type \
             -Werror=format-extra-args \
             -Wno-deprecated-declarations

CFLAGS = -std=gnu11 -fPIC -O2 $(WARN_FLAGS)

UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
  OPENCL_LIBS = -framework OpenCL
else
  OPENCL_FLAGS = -I$(PHONELIBS)/opencl/include
  OPENCL_LIBS = -L/system/vendor/lib64 -lgsl -lCB -lOpenCL
endif

OBJS += yuv_bench.o \
        ../../../common/util.o \
        ../../clutil.o

OUTPUT = yuv

.PHONY: all
all: $(OUTPUT)

$(OUTPUT): $(OBJS)
	@echo "[ LINK ] $@"
	$(CXX) -fPIC -o '$@' $^ \
        -L/usr/lib \
        $(OPENCL_LIBS)


%.o: %.cc
	@echo "[ CXX ] $@"
	$(CXX) $(CXXFLAGS) -MMD \
           -I../.. -I../../.. -I ../../../.. \
           $(OPENCL_FLAGS) \
           -c -o '$@' '$<'


%.o: %.c
	@echo "[ CC ] $@"
	$(CC) $(CFLAGS) -MMD \
          -I../.. -I../../.. -I ../../../.. \
          $(OPENCL_FLAGS) \
          -c -o '$@' '$<'

.PHONY: clean
clean:
	rm -f $(OUTPUT) $(OBJS) $(DEPS)

