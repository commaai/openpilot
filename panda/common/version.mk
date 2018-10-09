ifeq ($(RELEASE),1)
  BUILD_TYPE = "RELEASE"
else
  BUILD_TYPE = "DEBUG"
endif

ifneq ($(wildcard ../.git/HEAD),)
obj/gitversion.h: ../VERSION ../.git/HEAD ../.git/index
	echo "const uint8_t gitversion[] = \"$(shell cat ../VERSION)-$(shell git rev-parse --short=8 HEAD)-$(BUILD_TYPE)\";" > $@
else
ifneq ($(wildcard ../../.git/modules/panda/HEAD),)
obj/gitversion.h: ../VERSION ../../.git/modules/panda/HEAD ../../.git/modules/panda/index
	echo "const uint8_t gitversion[] = \"$(shell cat ../VERSION)-$(shell git rev-parse --short=8 HEAD)-$(BUILD_TYPE)\";" > $@
else
obj/gitversion.h: ../VERSION
	echo "const uint8_t gitversion[] = \"$(shell cat ../VERSION)-unknown-$(BUILD_TYPE)\";" > $@
endif
endif
