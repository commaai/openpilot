all: simple_kalman_impl.so

simple_kalman_impl.so: simple_kalman_impl.pyx simple_kalman_impl.pxd simple_kalman_setup.py
	python3 simple_kalman_setup.py build_ext --inplace
	rm -rf build
	rm simple_kalman_impl.c

.PHONY: clean
clean:
	rm -f simple_kalman_impl.so
