# Author : Byunghun Hwang <bh.hwang@iae.re.kr>
# Usage : make ARCH=armhf
# Note : You should make with GCC/G++ version 8

# Makefile
OS := $(shell uname)

#Set Architecutre
ARCH := armhf

CURRENT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
CURRENT_DIR_NAME := $(notdir $(patsubst %/,%,$(dir $(CURRENT_DIR))))

$(info $(CURRENT_DIR))


#Compilers
CC := g++
GCC := gcc
LD_LIBRARY_PATH += -L./lib/armhf
OUTDIR		= $(CURRENT_DIR)/../bin/
BUILDDIR		= $(CURRENT_DIR)/../bin/
INCLUDE_DIR = -I./ -I$(CURRENT_DIR)/ -I/usr/include/ -I/usr/include/opencv4/
LD_LIBRARY_PATH += -L/usr/local/lib -L./lib/armhf -L$(CURRENT_DIR)/../../lib/armhf/ -L./ -L$(CURRENT_DIR)/../bin/

# OS
ifeq ($(OS),Linux) #for Linux
	LDFLAGS = -Wl,--export-dynamic -Wl,-rpath=$(LD_LIBRARY_PATH)
	LDLIBS = -pthread -lrt -ldl -lm `pkg-config --cflags --libs opencv4`
	GTEST_LDLIBS = -lgtest
endif

$(shell mkdir -p $(OUTDIR))
$(shell mkdir -p $(BUILDDIR))

#if release(-O3), debug(-O0)
CXXFLAGS = -O3 -fPIC -Wall -std=c++17 -D__cplusplus=201703L

#custom definitions
CXXFLAGS += -D__MAJOR__=0 -D__MINOR__=0 -D__REV__=2
RM	= rm -rf

#directories
INCLUDE_FILES = . #$(CURRENT_DIR)/../../include
SOURCE_FILES = .

libfm.so : $(BUILDDIR)libfm.o
			  $(CC) $(LDFLAGS) $(LD_LIBRARY_PATH) -shared -o $(BUILDDIR)$@ $^ $(LDLIBS)

$(BUILDDIR)libfm.o: $(CURRENT_DIR)/libfm.cc
					   $(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $^ -o $@

test_fm : $(BUILDDIR)test_fm.o
			  $(CC) $(LDFLAGS) $(LD_LIBRARY_PATH) -o $(BUILDDIR)$@ $^ $(LDLIBS) -lfm
$(BUILDDIR)test_fm.o: $(CURRENT_DIR)/test_fm.cc
					   $(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $^ -o $@


all : test_fm libfm
clean : FORCE 
		$(RM) $(BUILDDIR)*.o
FORCE : 