# Author : Byunghun Hwang <bh.hwang@iae.re.kr>
# Usage : make ARCH=armhf
# Note : You should make with GCC/G++ version 8

# Makefile
OS := $(shell uname -s)
ARCH := $(shell uname -p)

PYTHON_VER := $(wordlist 2,4,$(subst ., ,$(shell python3 -V 2>&1)))
PYTHON_VER_MAJOR := $(word 1,${PYTHON_VER})
PYTHON_VER_MINOR := $(word 2,${PYTHON_VER})
PYTHON_VER_REV := $(word 3,${PYTHON_VER})

$(info $(CURRENT_DIR))

CURRENT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
CURRENT_DIR_NAME := $(notdir $(patsubst %/,%,$(dir $(CURRENT_DIR))))

$(info Installed Python3 Version : $(shell python3 -V))


#Compilers
CC := g++
GCC := gcc
LD_LIBRARY_PATH += -L./lib/armhf
OUTDIR		= $(CURRENT_DIR)/../bin/
BUILDDIR		= $(CURRENT_DIR)/../bin/
INCLUDE_DIR = -I./ -I$(CURRENT_DIR)/ -I/usr/include/ -I/usr/include/opencv4/
LD_LIBRARY_PATH += -L/usr/local/lib -L./lib/armhf -L$(CURRENT_DIR)/../../lib/armhf/ -L./ -L$(CURRENT_DIR)/../bin/


LDFLAGS = -Wl,--export-dynamic -Wl,-rpath=$(LD_LIBRARY_PATH)
LDLIBS = -pthread -lrt -ldl -lm $(shell pkg-config --cflags --libs opencv4)
CXXFLAGS = -O3 -fPIC -Wall -std=c++17 -D__cplusplus=201703L
# OS
ifeq ($(OS),Linux) #for Linux
	ifeq ($(PYTHON_VER_MINOR),10)
		INCLUDE_DIR += -I/usr/include/python3.10/
		LDLIBS += $(shell python3.10-config --libs --embed) -lpython3.10
		CXXFLAGS += $(shell python3.10-config --cflags)
	else ifeq ($(PYTHON_VER_MINOR),9)
		INCLUDE_DIR += -I/usr/include/python3.9/
		LDLIBS += $(shell python3.9-config --libs --embed) -lpython3.9
		CXXFLAGS += $(shell python3.9-config --cflags)
	else ifeq ($(PYTHON_VER_MINOR),8)
		INCLUDE_DIR += -I/usr/include/python3.8/
		LDLIBS += $(shell python3.8-config --libs --embed) -lpython3.8
		CXXFLAGS += $(shell python3.8-config --cflags)
	endif
endif

$(shell mkdir -p $(OUTDIR))
$(shell mkdir -p $(BUILDDIR))

#if release(-O3), debug(-O0)


#custom definitions
CXXFLAGS += -D__MAJOR__=0 -D__MINOR__=0 -D__REV__=1 -D__PYTHON_VER_MAJOR__=$(PYTHON_VER_MAJOR) -D__PYTHON_VER_MINOR__=$(PYTHON_VER_MINOR)
RM	= rm -rf

#directories
INCLUDE_FILES = . #$(CURRENT_DIR)/../../include
SOURCE_FILES = .

libppe.so : $(BUILDDIR)libppe.o
			  $(CC) $(LDFLAGS) $(LD_LIBRARY_PATH) -shared -o $(BUILDDIR)$@ $^ $(LDLIBS)

$(BUILDDIR)libppe.o: $(CURRENT_DIR)/libppe.cc
					   $(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $^ -o $@

test_ppe : $(BUILDDIR)test_ppe.o
			  $(CC) $(LDFLAGS) $(LD_LIBRARY_PATH) -o $(BUILDDIR)$@ $^ $(LDLIBS) -lppe
$(BUILDDIR)test_ppe.o: $(CURRENT_DIR)/test_ppe.cc
					   $(CC) $(CXXFLAGS) $(INCLUDE_DIR) -c $^ -o $@


all : test_ppe libppe
clean : FORCE 
		$(RM) $(BUILDDIR)*.o
FORCE : 