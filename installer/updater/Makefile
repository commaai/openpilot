CC = clang
CXX = clang++

PHONELIBS = ../../phonelibs

WARN_FLAGS = -Werror=implicit-function-declaration \
             -Werror=incompatible-pointer-types \
             -Werror=int-conversion \
             -Werror=return-type \
             -Werror=format-extra-args

CFLAGS = -std=gnu11 -g -fPIC -O2 $(WARN_FLAGS)
CXXFLAGS = -std=c++11 -g -fPIC -O2 $(WARN_FLAGS)

CURL_FLAGS = -I$(PHONELIBS)/curl/include
CURL_LIBS = $(PHONELIBS)/curl/lib/libcurl.a \
            $(PHONELIBS)/zlib/lib/libz.a

BORINGSSL_FLAGS = -I$(PHONELIBS)/boringssl/include
BORINGSSL_LIBS = $(PHONELIBS)/boringssl/lib/libssl_static.a \
                 $(PHONELIBS)/boringssl/lib/libcrypto_static.a \

NANOVG_FLAGS = -I$(PHONELIBS)/nanovg

JSON11_FLAGS = -I$(PHONELIBS)/json11

OPENGL_LIBS = -lGLESv3

FRAMEBUFFER_LIBS = -lutils -lgui -lEGL

.PHONY: all
all: updater

OBJS = opensans_regular.ttf.o \
			 opensans_semibold.ttf.o \
			 opensans_bold.ttf.o \
       ../../selfdrive/common/touch.o \
       ../../selfdrive/common/framebuffer.o \
       $(PHONELIBS)/json11/json11.o \
       $(PHONELIBS)/nanovg/nanovg.o

DEPS := $(OBJS:.o=.d)

updater: updater.o $(OBJS)
	@echo "[ LINK ] $@"
	$(CXX) $(CPPFLAGS) -fPIC -o 'updater' $^ \
    $(FRAMEBUFFER_LIBS) \
    $(CURL_LIBS) \
    $(BORINGSSL_LIBS) \
    -L/system/vendor/lib64 \
    $(OPENGL_LIBS) \
    -lcutils -lm -llog
	strip updater

opensans_regular.ttf.o: ../../selfdrive/assets/fonts/opensans_regular.ttf
	@echo "[ bin2o ] $@"
	cd '$(dir $<)' && ld -r -b binary '$(notdir $<)' -o '$(abspath $@)'

opensans_bold.ttf.o: ../../selfdrive/assets/fonts/opensans_bold.ttf
	@echo "[ bin2o ] $@"
	cd '$(dir $<)' && ld -r -b binary '$(notdir $<)' -o '$(abspath $@)'

opensans_semibold.ttf.o: ../../selfdrive/assets/fonts/opensans_semibold.ttf
	@echo "[ bin2o ] $@"
	cd '$(dir $<)' && ld -r -b binary '$(notdir $<)' -o '$(abspath $@)'

%.o: %.c
	mkdir -p $(@D)
	@echo "[ CC ] $@"
	$(CC) $(CPPFLAGS) $(CFLAGS) \
          -I../.. \
          -I$(PHONELIBS)/android_frameworks_native/include \
          -I$(PHONELIBS)/android_system_core/include \
          -I$(PHONELIBS)/android_hardware_libhardware/include \
          $(NANOVG_FLAGS) \
          -c -o '$@' '$<'

%.o: %.cc
	mkdir -p $(@D)
	@echo "[ CXX ] $@"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) \
           -I../../selfdrive \
           -I../../ \
           -I$(PHONELIBS)/android_frameworks_native/include \
           -I$(PHONELIBS)/android_system_core/include \
           -I$(PHONELIBS)/android_hardware_libhardware/include \
           $(NANOVG_FLAGS) \
           $(JSON11_FLAGS) \
           $(CURL_FLAGS) \
           $(BORINGSSL_FLAGS) \
           -c -o '$@' '$<'


.PHONY: clean
clean:
	rm -f $(OBJS) $(DEPS)

-include $(DEPS)
