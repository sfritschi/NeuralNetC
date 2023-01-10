CC=cc
CFLAGS=-Wall -Wextra -Wpedantic -ggdb3 -D_GNU_SOURCE -std=c17

INCLUDE=include
SRC_DIR=src
TARGET=main
.PHONY: all, clean
all:$(TARGET)

$(TARGET): $(SRC_DIR)/$(TARGET).c
	$(CC) $(CFLAGS) -I$(INCLUDE) -o $@ $< -lm

clean:
	$(RM) $(TARGET)
