CC = clang
CXX = clang++

PHONELIBS = ../../../../phonelibs
COMMON = ../../../common

WARN_FLAGS = -Werror=implicit-function-declaration \
             -Werror=incompatible-pointer-types \
             -Werror=int-conversion \
             -Werror=return-type \
             -Werror=format-extra-args

CFLAGS = -std=gnu11 -fPIC -O2 $(WARN_FLAGS)
CXXFLAGS = -std=c++1z -fPIC -O2 $(WARN_FLAGS)

NANOVG_FLAGS = -I$(PHONELIBS)/nanovg

OPENGL_LIBS = -lGLESv3

FRAMEBUFFER_LIBS = -lutils -lgui -lEGL

OBJS = text.o \
       $(COMMON)/framebuffer.o \
       $(COMMON)/util.o \
       $(COMMON)/touch.o \
       $(PHONELIBS)/nanovg/nanovg.o \
       opensans_regular.o \

DEPS := $(OBJS:.o=.d)

.PHONY: all
all: text

text: $(OBJS)
	@echo "[ LINK ] $@"
	$(CXX) -fPIC -o '$@' $^ \
	      -s \
        $(FRAMEBUFFER_LIBS) \
        -L/system/vendor/lib64 \
        $(OPENGL_LIBS) \
        -lm -llog

opensans_regular.o: ../../../assets/fonts/opensans_regular.ttf
	@echo "[ bin2o ] $@"
	cd '$(dir $<)' && ld -r -b binary '$(notdir $<)' -o '$(abspath $@)'

%.o: %.cc
	mkdir -p $(@D)
	@echo "[ CXX ] $@"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
           -I../../../selfdrive \
           -I../../../ \
           -I$(PHONELIBS)/android_frameworks_native/include \
           -I$(PHONELIBS)/android_system_core/include \
           -I$(PHONELIBS)/android_hardware_libhardware/include \
           $(NANOVG_FLAGS) \
           -c -o '$@' '$<'


.PHONY: clean
clean:
	rm -f text $(OBJS) $(DEPS)

-include $(DEPS)
