compile = gcc
objflags = -g -Wall -Wextra  -c
linkflags = -g -Wall -Wextra 

all: count3
	$(compile) -fopenmp count3.o -o program -pthread


count3: count3.c
	$(compile)  -fopenmp  $(objflags) count3.c

clean:
	rm -f *.o *~ count3
