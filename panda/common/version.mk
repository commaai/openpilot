ifeq ($(RELEASE),1)
	BUILD_TYPE = "RELEASE"
else
	BUILD_TYPE = "DEBUG"
endif

SELF_DIR := $(dir $(lastword $(MAKEFILE_LIST)))

ifneq ($(wildcard $(SELF_DIR)/../.git/HEAD),)
obj/gitversion.h: $(SELF_DIR)/../VERSION $(SELF_DIR)/../.git/HEAD $(SELF_DIR)/../.git/index
	echo "const uint8_t gitversion[] = \"$(shell cat $(SELF_DIR)/../VERSION)-$(BUILDER)-$(shell git rev-parse --short=8 HEAD)-$(BUILD_TYPE)\";" > $@
else
ifneq ($(wildcard $(SELF_DIR)/../../.git/modules/panda/HEAD),)
obj/gitversion.h: $(SELF_DIR)/../VERSION $(SELF_DIR)/../../.git/modules/panda/HEAD $(SELF_DIR)/../../.git/modules/panda/index
	echo "const uint8_t gitversion[] = \"$(shell cat $(SELF_DIR)/../VERSION)-$(BUILDER)-$(shell git rev-parse --short=8 HEAD)-$(BUILD_TYPE)\";" > $@
else
obj/gitversion.h: $(SELF_DIR)/../VERSION
	echo "const uint8_t gitversion[] = \"$(shell cat $(SELF_DIR)/../VERSION)-$(BUILDER)-unknown-$(BUILD_TYPE)\";" > $@
endif
endif
