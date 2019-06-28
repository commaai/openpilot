CC = clang
CCFLAGS = -O3 -fPIC -DPANDA -I.

.PHONY: all
all: libpandasafety.so

libpandasafety.so: test.o
	$(CC) -shared -o '$@' $^ -lm

test.o: test.c
	@echo "[ CC ] $@"
	$(CC) $(CCFLAGS) -MMD -c -I../../board -o '$@' '$<'

.PHONY: clean
clean:
	rm -f libpandasafety.so test.o test.d

-include test.d
