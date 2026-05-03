CC_NATIVE  = gcc
CC_ARM     = arm-none-eabi-gcc
CFLAGS     = -O2 -Wall -Wextra -std=c99

.PHONY: native arm clean

native: ascon128.c test_vectors.c ascon128.h
	$(CC_NATIVE) $(CFLAGS) -o test_native ascon128.c test_vectors.c
	./test_native

arm: ascon128.c ascon128.h
	$(CC_ARM) $(CFLAGS) -mcpu=cortex-m3 -mthumb -c -o ascon128.o ascon128.c
	$(CC_ARM) -mcpu=cortex-m3 -mthumb -nostdlib \
	    -Wl,--entry=ascon128_encrypt -o ascon128.elf ascon128.o 2>/dev/null || \
	    mv ascon128.o ascon128.elf
	@echo "ARM ELF produced: ascon128.elf"
	@file ascon128.elf || true

clean:
	rm -f test_native ascon128.o ascon128.elf
