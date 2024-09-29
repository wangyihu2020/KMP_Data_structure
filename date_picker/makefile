##
## export SDK_PATH=/home/b...
##

include $(NVR_ROOT_PATH)/Makefile.param

.PHONY: all target clean

all:target


target:
	@echo "get ready for build calendar."
	@cd ../;ln -snf $(SDK_PATH)/awtkcode/awtk ./awtk
	@cd ../;ln -snf $(SDK_PATH)/awtkcode/awtk-linux-fb ./awtk-linux-fb
	scons LINUX_FB=TRUE;
	cp bin/*.so $(NVR_ROOT_PATH)/opt/lib

clean:
	scons -c;
