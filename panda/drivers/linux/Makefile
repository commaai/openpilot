VERSION=0.0.1
obj-m+=panda.o

all:	build install

build:
	sudo dkms build panda/$(VERSION)

install:
	sudo dkms install panda/$(VERSION)

remove:
	sudo dkms remove panda/$(VERSION) --all

uninstall:
	sudo dkms uninstall panda/$(VERSION)

clean:	remove

link:
	sudo dkms add `pwd`

unload:
	sudo rmmod panda
