ECOSDIR=../../extern/ecos/ecos-1.0.4
FMIinclude=../../extern/fmi

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

OBJ = polytope.o FSM.o mealydata.o pppdata.o TuLiPControl.o data.o

# FILES TO CLEAN UP
CLEAN = *.o 

C = $(CC) $(CFLAGS) -Iinclude -I$(ECOSDIR)/external/ldl/include -I$(ECOSDIR)/external/amd/include -I$(ECOSDIR)/external/SuiteSparse_config -I$(ECOSDIR)/include -I$(FMIinclude)

all: fmu ecos test obj

# build the ecos library 
ecos:
	(cd $(ECOSDIR)	; $(MAKE) ldl amd ecos)

# builde the .o files

obj: polytope.o FSM.o mealydata.o pppdata.o TuLiPControl.o data.o

polytope.o: include/polytope.h sources/polytope.c 
	$(C) -c sources/polytope.c -o polytope.o

FSM.o: include/FSM.h sources/FSM.c include/mealydata.h
	$(C) -c sources/FSM.c -o FSM.o

mealydata.o: include/mealydata.h sources/mealydata.c 
	$(C) -c sources/mealydata.c -o mealydata.o

pppdata.o: include/pppdata.h sources/pppdata.c 
	$(C) -c sources/pppdata.c -o pppdata.o

TuLiPControl.o: sources/TuLiPControl.c include/TuLiPControl.h include/polytope.h include/data.h include/pppdata.h include/FSM.h include/mealydata.h
	$(C) -c sources/TuLiPControl.c -o TuLiPControl.o

data.o: sources/data.c include/data.h include/polytope.h 
	$(C) -c sources/data.c -o data.o

fmu:ecos obj
	mkdir -p binaries/$(ARCH)
ifeq ($(ARCH), darwin64)
	$(C) -DFMI_COSIMULATION -shared -Wl,-dylib -L$(ECOSDIR) sources/TuLiPFMU.c $(OBJ) -lecos $(LIBS) -o binaries/$(ARCH)/TuLiPFMU.dylib
else
	$(C) -DFMI_COSIMULATION -shared -Wl,-soname,TuLiPFMU.so -L$(ECOSDIR) sources/TuLiPFMU.c $(OBJ) -lecos $(LIBS) -o binaries/$(ARCH)/TuLiPFMU.so
endif
	zip -r TuLiPFMU.fmu binaries/ include/ sources/ modelDescription.xml 

test: testController testFSM

testFSM: ecos obj test/testFSM.c 
	$(C) -o testFSM test/testFSM.c FSM.o mealydata.o $(LIBS)
	echo testFSM successfully built. Type ./testFSM to run 

testController: ecos obj test/testController.c 
	$(C) -L$(ECOSDIR) test/testController.c $(ECOSDIR)/libecos.a $(OBJ) -lecos $(LIBS) -o testController 
	echo testController successfully built. Type ./testController to run 

# remove object files, but keep the compiled programs 
clean:
	( cd $(ECOSDIR)    ; $(MAKE) clean )
	- $(RM) $(CLEAN)

# clean, and then remove compiled programs 
purge: clean
	( cd $(ECOSDIR)    ; $(MAKE) purge )
	- $(RM) testFSM testController TuLiPFMU.fmu

