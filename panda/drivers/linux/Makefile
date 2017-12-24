VERSION=0.0.1
obj-m+=panda.o

link:
	sudo dkms add `pwd`

build:
	sudo dkms build panda/$(VERSION)

install:
	sudo dkms install panda/$(VERSION)

all:	build install

uninstall:
	sudo dkms uninstall panda/$(VERSION)
	sudo dkms remove panda/$(VERSION) --all

