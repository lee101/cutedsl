CC ?= gcc
CFLAGS = -O3 -march=native -Wall -Wextra -fopenmp
LDFLAGS = -lm -fopenmp

.PHONY: all test clean

all: test_kernels

test_kernels: test_kernels.c rms_norm.h silu_gate.h chronos_preprocess.h chronos_preprocess.c
	$(CC) $(CFLAGS) -o $@ test_kernels.c chronos_preprocess.c $(LDFLAGS)

test: test_kernels
	./test_kernels

clean:
	rm -f test_kernels *.o
