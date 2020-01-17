CC := gcc

vidindex: bitstream.c bitstream.h vidindex.c
	$(eval $@_TMP := $(shell mktemp))
	$(CC) -std=c99 bitstream.c vidindex.c -o $($@_TMP)
	mv $($@_TMP) $@
