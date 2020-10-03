CC = clang
CXX = clang++


PHONELIBS = ../phonelibs

WARN_FLAGS = -Werror=implicit-function-declaration \
             -Werror=incompatible-pointer-types \
             -Werror=int-conversion \
             -Werror=return-type \
             -Werror=format-extra-args

CFLAGS = -std=gnu11 -fPIC -O2 $(WARN_FLAGS)
CXXFLAGS = -std=c++11 -fPIC -O2 $(WARN_FLAGS)

NANOVG_FLAGS = -I$(PHONELIBS)/nanovg

OPENGL_LIBS = -lGLESv3

FRAMEBUFFER_LIBS = -lutils -lgui -lEGL

COMMON_OBJS = ../selfdrive/common/glutil.o \
       ../selfdrive/common/framebuffer.o \
       ../selfdrive/common/util.o \
       $(PHONELIBS)/nanovg/nanovg.o \
       ../selfdrive/common/spinner.o \
       opensans_semibold.o \
       img_spinner_track.o \
       img_spinner_comma.o

OPENPILOT_OBJS = installer_openpilot.o \
       continue_openpilot.o \
			 $(COMMON_OBJS)

DASHCAM_OBJS = installer_dashcam.o \
       continue_dashcam.o \
			 $(COMMON_OBJS)

DEPS := $(OPENPILOT_OBJS:.o=.d) $(DASHCAM_OBJS:.o=.d)

.PHONY: all
all: installers/installer_dashcam installers/installer_openpilot

installers/installer_openpilot: $(OPENPILOT_OBJS)
	@echo "[ LINK ] $@"
	$(CXX) -fPIC -o '$@' $^ \
	      -s \
        $(FRAMEBUFFER_LIBS) \
        -L/system/vendor/lib64 \
        $(OPENGL_LIBS) \
        -lm -llog

installers/installer_dashcam: $(DASHCAM_OBJS)
	@echo "[ LINK ] $@"
	$(CXX) -fPIC -o '$@' $^ \
	      -s \
        $(FRAMEBUFFER_LIBS) \
        -L/system/vendor/lib64 \
        $(OPENGL_LIBS) \
        -lm -llog

../selfdrive/common/framebuffer.o: ../selfdrive/common/framebuffer.cc
	@echo "[ CXX ] $@"
	$(CXX) $(CXXFLAGS) -MMD \
           -I$(PHONELIBS)/android_frameworks_native/include \
           -I$(PHONELIBS)/android_system_core/include \
           -I$(PHONELIBS)/android_hardware_libhardware/include \
           -c -o '$@' '$<'

../selfdrive/common/spinner.o: ../selfdrive/common/spinner.c
	@echo "[ CXX ] $@"
	$(CXX) $(CXXFLAGS) -MMD \
          -I$(PHONELIBS)/android_frameworks_native/include \
          -I$(PHONELIBS)/android_system_core/include \
          -I$(PHONELIBS)/android_hardware_libhardware/include \
          $(NANOVG_FLAGS) \
          -c -o '$@' '$<'

opensans_semibold.o: ../selfdrive/assets/fonts/opensans_semibold.ttf
	@echo "[ bin2o ] $@"
	cd '$(dir $<)' && ld -r -b binary '$(notdir $<)' -o '$(abspath $@)'

img_spinner_track.o: ../selfdrive/assets/img_spinner_track.png
	@echo "[ bin2o ] $@"
	cd '$(dir $<)' && ld -r -b binary '$(notdir $<)' -o '$(abspath $@)'

img_spinner_comma.o: ../selfdrive/assets/img_spinner_comma.png
	@echo "[ bin2o ] $@"
	cd '$(dir $<)' && ld -r -b binary '$(notdir $<)' -o '$(abspath $@)'

installer_openpilot.o: installer.c
	@echo "[ CC ] $@"
	$(CC) $(CFLAGS) -MMD \
          -I.. -I../selfdrive \
          -DBRAND=openpilot \
					-DBRANCH=release2 \
          -c -o '$@' '$<'

installer_dashcam.o: installer.c
	@echo "[ CC ] $@"
	$(CC) $(CFLAGS) -MMD \
          -I.. -I../selfdrive \
          -DBRAND=dashcam \
					-DBRANCH=dashcam \
          -c -o '$@' '$<'

%.o: %.c
	@echo "[ CC ] $@"
	$(CC) $(CFLAGS) -MMD \
          -I.. -I../selfdrive \
          -c -o '$@' '$<'

%.o: %.sh
	@echo "[ bin2o ] $@"
	cd '$(dir $<)' && ld -r -b binary '$(notdir $<)' -o '$(abspath $@)'

.PHONY: clean
clean:
	rm -f installer_openpilot installer_dashcam $(OPENPILOT_OBJS) $(DASHCAM_OBJS) $(DEPS)

-include $(DEPS)
