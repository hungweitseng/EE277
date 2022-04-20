CONFIG_MODULE_SIG=n


ifdef SYSSRC
 KERNEL_SOURCES	 = $(SYSSRC)
else
 KERNEL_UNAME	:= $(shell uname -r)
 KERNEL_SOURCES	 = /lib/modules/$(KERNEL_UNAME)/build
endif

obj-y		+= host/
obj-y		+= target/

#NVIDIA_SRC := /usr/src/nvidia-450.51.06/
default: modules
.PHONY: default
install: modules_install
	update-initramfs -u
#	cp host/nv_pinbuf.h /usr/include/nvmed

.PHONY: install


#nvidia-syms:
#	make -C $(KERNEL_SOURCES) SUBDIRS=$(NVIDIA_SRC) modules

.PHONY:


%::
	$(MAKE) -C $(KERNEL_SOURCES) \
        M=$$PWD $@
#        KBUILD_EXTRA_SYMBOLS=$(NVIDIA_SRC)/Module.symvers \
#	    NVIDIA_SRC=$(NVIDIA_SRC) \
