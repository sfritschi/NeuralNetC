CC=clang
CFLAGS=-Wall -Wextra -Wpedantic -ggdb3 -D_GNU_SOURCE -std=c17

INCLUDE=include
SRC_DIR=src
TARGET=main
.PHONY: all, release, clean
all:$(TARGET)

$(TARGET): $(SRC_DIR)/$(TARGET).c
	$(CC) $(CFLAGS) -I$(INCLUDE) -o $@ $< -lm

release: $(SRC_DIR)/$(TARGET).c
	$(CC) $(CFLAGS) -O3 -march=native -DNDEBUG -I$(INCLUDE) -o $@ $< -lm
	 
clean:
	$(RM) $(TARGET)
	$(RM) release
