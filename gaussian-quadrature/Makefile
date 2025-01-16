CC=gcc
CFLAGS=-Wall -Wextra -O2 -I.
DEPS = gaussian_quadrature.h
OBJ = gaussian_quadrature.o main.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

gaussian_quadrature: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) -lm

.PHONY: clean

clean:
	rm -f *.o gaussian_quadrature
