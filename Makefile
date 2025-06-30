##################################################
# PROJECT: DXL Protocol 2.0 Read/Write Example Makefile
# AUTHOR : ROBOTIS Ltd.
##################################################

#---------------------------------------------------------------------
# Makefile template for projects using DXL SDK
#
# Please make sure to follow these instructions when setting up your
# own copy of this file:
#
#   1- Enter the name of the target (the TARGET variable)
#   2- Add additional source files to the SOURCES variable
#   3- Add additional static library objects to the OBJECTS variable
#      if necessary
#   4- Ensure that compiler flags, INCLUDES, and LIBRARIES are
#      appropriate to your needs
#
#
# This makefile will link against several libraries, not all of which
# are necessarily needed for your project.  Please feel free to
# remove libaries you do not need.
#---------------------------------------------------------------------

# *** ENTER THE TARGET NAME HERE ***
TARGET      = play_audio_DXL

# important directories used by assorted rules and other variables
DIR_DXL    = /home/hyeonwoo/DXL/DynamixelSDK-3.7.31/c++
DIR_OBJS   = .objects

# compiler options
CC          = gcc
CX          = g++
CCFLAGS     = -O2 -O3 -DLINUX -D_GNU_SOURCE -Wall $(INCLUDES) -g
CXFLAGS     = -O2 -O3 -DLINUX -D_GNU_SOURCE -Wall $(INCLUDES) -g
LNKCC       = $(CX)
LNKFLAGS    = $(CXFLAGS) #-Wl,-rpath,$(DIR_THOR)/lib
#FORMAT      = -m64

#---------------------------------------------------------------------
# Core components (all of these are likely going to be needed)
#---------------------------------------------------------------------
INCLUDES   += -I/home/hyeonwoo/DXL/DynamixelSDK-3.7.31/c++/include/dynamixel_sdk
INCLUDES   += -I/usr/include/eigen3
INCLUDES   += -I/usr/include/opencv4
INCLUDES   += -I/usr/include/SFML
INCLUDES += -I$(CURDIR)
INCLUDES += -I$(CURDIR)/cnpy
INCLUDES += -I$(CURDIR)/cnpy/zlib


LNKFLAGS += -L/home/hyeonwoo/DXL/DynamixelSDK-3.7.31/c++/build/linux64
LIBRARIES += -ldxl_x64_cpp -lrt -lasound -lsndfile -lportaudio -lsfml-audio -lsfml-system -lz



#---------------------------------------------------------------------
# Files
#---------------------------------------------------------------------
SOURCES = play_audio_DXL.cpp
MACRO = Macro_function.cpp cnpy/cnpy.cpp
    # *** OTHER SOURCES GO HERE ***

OBJECTS  = $(addsuffix .o,$(addprefix $(DIR_OBJS)/,$(basename $(notdir $(SOURCES)))))
#OBJETCS += *** ADDITIONAL STATIC LIBRARIES GO HERE ***
OBJECTS  += $(addsuffix .o,$(addprefix $(DIR_OBJS)/,$(basename $(notdir $(MACRO)))))



#---------------------------------------------------------------------
# Compiling Rules
#---------------------------------------------------------------------
$(TARGET): make_directory $(OBJECTS)
	$(LNKCC) $(LNKFLAGS) $(OBJECTS) -o $(TARGET) $(LIBRARIES)
 
all: $(TARGET)

clean:
	rm -rf $(TARGET) $(DIR_OBJS) core *~ *.a *.so *.lo

make_directory:
	mkdir -p $(DIR_OBJS)/

$(DIR_OBJS)/%.o: %.cpp
	$(CX) $(CXFLAGS) -c $< -o $@
$(DIR_OBJS)/%.o: cnpy/%.cpp
	$(CX) $(CXFLAGS) -c $< -o $@
#---------------------------------------------------------------------
# End of Makefile
#---------------------------------------------------------------------
