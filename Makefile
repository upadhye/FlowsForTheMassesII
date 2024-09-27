CCFLAGS=-O3 -fopenmp -Wno-unused-result
PATHS=-I/usr/include -I/usr/lib/gcc/x86_64-linux-gnu/6/include/ -L/usr/lib/x86_64-linux-gnu/
LIBS=-lgsl -lgslcblas -lm 

FlowsForTheMassesII: FlowsForTheMassesII.cc Makefile AU_tabfun.h AU_ncint.h AU_fftgrid.h AU_cosmoparam.h AU_cosmofunc.h AU_CMB_lensing.h AU_fastpt_coord.h AU_combinatorics.h AU_fluid.h 
	g++ FlowsForTheMassesII.cc -o FlowsForTheMassesII $(CCFLAGS) $(PATHS) $(LIBS) 

clean:
	$(RM) FlowsForTheMassesII

