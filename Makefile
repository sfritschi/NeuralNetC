CC=cc
CFLAGS=-Wall -Wextra -Wpedantic -g -std=c17

SRC_DIR=src
TARGET=main
.PHONY: all, clean
all:$(TARGET)

$(TARGET): $(SRC_DIR)/$(TARGET).c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	$(RM) $(TARGET)
