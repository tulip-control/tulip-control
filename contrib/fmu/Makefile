ECOSDIR=../../extern/ecos/ecos-1.0.4
FMIinclude=../../extern/fmi

BUILD=build/

CFLAGS = -O2 -Wall -DLDL_LONG -DDLONG -Wextra -fPIC
LIBS = -lm

# Try to determine architecture if user did not specify it.
ifndef $(ARCH)
ifeq ($(shell uname), Linux)
ARCH=linux64
else
ARCH=darwin64
endif
endif

# rt is linked in the same manner as in the Makefile of ecos.
# For Linux `-lrt` may be required with clock_gettime().
ifeq ($(shell uname), Linux)
LIBS += -lrt
endif

OBJ = $(BUILD)polytope.o $(BUILD)FSM.o $(BUILD)mealydata.o $(BUILD)pppdata.o $(BUILD)TuLiPControl.o $(BUILD)data.o

# Files to clean up
CLEAN = *.o

C = $(CC) $(CFLAGS) -Iinclude -I$(ECOSDIR)/external/ldl/include -I$(ECOSDIR)/external/amd/include -I$(ECOSDIR)/external/SuiteSparse_config -I$(ECOSDIR)/include -I$(FMIinclude)

all: fmu ecos test obj

# build the ecos library
ecos:
	(cd $(ECOSDIR)	; $(MAKE) ldl amd ecos)

# build the .o files

# Keep in sync with OBJ variable
obj: polytope.o FSM.o mealydata.o pppdata.o TuLiPControl.o data.o

polytope.o: include/polytope.h sources/polytope.c
	$(C) -c sources/polytope.c -o $(BUILD)polytope.o

FSM.o: include/FSM.h sources/FSM.c include/mealydata.h
	$(C) -c sources/FSM.c -o $(BUILD)FSM.o

mealydata.o: include/mealydata.h sources/mealydata.c
	$(C) -c sources/mealydata.c -o $(BUILD)mealydata.o

pppdata.o: include/pppdata.h sources/pppdata.c
	$(C) -c sources/pppdata.c -o $(BUILD)pppdata.o

TuLiPControl.o: sources/TuLiPControl.c include/TuLiPControl.h include/polytope.h include/data.h include/pppdata.h include/FSM.h include/mealydata.h
	$(C) -c sources/TuLiPControl.c -o $(BUILD)TuLiPControl.o

data.o: sources/data.c include/data.h include/polytope.h
	$(C) -c sources/data.c -o $(BUILD)data.o

fmu:ecos obj
	mkdir -p binaries/$(ARCH)
ifeq ($(ARCH), darwin64)
	$(C) -DFMI_COSIMULATION -shared -Wl,-dylib -L$(ECOSDIR) sources/TuLiPFMU.c $(OBJ) -lecos $(LIBS) -o binaries/$(ARCH)/TuLiPFMU.dylib
else
	$(C) -DFMI_COSIMULATION -shared -Wl,-soname,TuLiPFMU.so -L$(ECOSDIR) sources/TuLiPFMU.c $(OBJ) -lecos $(LIBS) -o binaries/$(ARCH)/TuLiPFMU.so
endif
	zip -r TuLiPFMU.fmu binaries/ include/ sources/ modelDescription.xml

test: test_controller test_fsm

test_fsm: ecos obj test/test_fsm.c
	$(C) -o $(BUILD)test_fsm test/test_fsm.c $(BUILD)FSM.o $(BUILD)mealydata.o $(LIBS)
	@echo test_fsm successfully built. Type $(BUILD)test_fsm to run

test_controller: ecos obj test/test_controller.c
	$(C) -L$(ECOSDIR) test/test_controller.c $(ECOSDIR)/libecos.a $(OBJ) -lecos $(LIBS) -o $(BUILD)test_controller
	@echo test_controller successfully built. Type $(BUILD)test_controller to run

# remove object files, but keep the compiled programs
clean:
	( cd $(ECOSDIR)    ; $(MAKE) clean )
	- $(RM) $(BUILD)$(CLEAN) modelDescription.xml sources/data.c sources/mealydata.c sources/pppdata.c include/TuLiPFMU.h include/mealydata.h
	- $(RM) $(BUILD)*.p
	- rm -rf binaries

# clean, and then remove compiled programs
purge: clean
	( cd $(ECOSDIR)    ; $(MAKE) purge )
	- $(RM) $(BUILD)test_fsm $(BUILD)test_controller TuLiPFMU.fmu
