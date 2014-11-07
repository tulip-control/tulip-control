CFLAGS = -O2 -Wall -DLDL_LONG -DDLONG -Wextra -fPIC

LIBS = -lm -lrt

OBJ = polytope.o FSM.o mealydata.o pppdata.o TuLiPControl.o data.o

# FILES TO CLEAN UP
CLEAN = *.o 

C = $(CC) $(CFLAGS) -Iinclude -I../../extern/ecos-1.0.4/external/ldl/include -I../../extern/ecos-1.0.4/external/amd/include -I../../extern/ecos-1.0.4/external/SuiteSparse_config -I../../extern/ecos-1.0.4/include -I../../extern/fmi

all: fmu ecos test obj

# build the ecos library 
ecos:
	(cd ../../extern/ecos-1.0.4	; $(MAKE) ldl amd ecos)

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
	mkdir -p binaries/linux64
	$(C) -DFMI_COSIMULATION -c sources/TuLiPFMU.c -o TuLiPFMU.o 
	$(C) -DFMI_COSIMULATION -shared -Wl,-soname,TuLiPFMU.so -L../../extern/ecos-1.0.4 TuLiPFMU.o $(OBJ) -lecos $(LIBS) -o binaries/linux64/TuLiPFMU.so 
	zip -r TuLiPFMU.fmu binaries/ include/ sources/ modelDescription.xml 

test: testController testFSM

testFSM: ecos obj test/testFSM.c 
	$(C) -o testFSM test/testFSM.c FSM.o mealydata.o $(LIBS)
	echo testFSM successfully built. Type ./testFSM to run 

testController: ecos obj test/testController.c 
	$(C) -L../../extern/ecos-1.0.4 test/testController.c ../../extern/ecos-1.0.4/libecos.a $(OBJ) -lecos $(LIBS) -o testController 
	echo testController successfully built. Type ./testController to run 

# remove object files, but keep the compiled programs 
clean:
	( cd ../../extern/ecos-1.0.4    ; $(MAKE) clean )
	- $(RM) $(CLEAN)

# clean, and then remove compiled programs 
purge: clean
	( cd ../../extern/ecos-1.0.4    ; $(MAKE) purge )
	- $(RM) testFSM testController TuLiPFMU.fmu

