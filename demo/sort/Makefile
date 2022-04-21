all: arraySortAA binary arraySort print gen

EXTRA_CFLAGS = -O3 -mavx  -g

LD_FLAGS = -lpthread -pthread -lm

arraySort: arraySort.cpp 
	g++ -g -O3 arraySort.cpp -o arraySort $(LD_FLAGS)


arraySortAA: arraySortAA.c aasort.o
	gcc  $(EXTRA_CFLAGS) -DAASORT aasort.o arraySortAA.c -o arraySortAA $(LD_FLAGS)

binary: binary.c 
	gcc  $(EXTRA_CFLAGS) aasort.c binary.c -o binary $(LD_FLAGS)

aasort.o: aasort.c
	gcc  $(EXTRA_CFLAGS) aasort.c -c $(LD_FLAGS)

print: print.c 
	gcc  $(EXTRA_CFLAGS) print.c -o print $(LD_FLAGS)

gen: gen.c 
	gcc  $(EXTRA_CFLAGS) gen.c -o gen $(LD_FLAGS)

clean:
		rm -f arraySortAA arraySort  binary print gen *~ *.o
