CC := g++

INCLUDES := -I/usr/local/include/opencv
CFLAGS := -g -Wall
LFLAGS := -L/usr/local/lib -lcv -lhighgui
BIN := main

all: $(BIN)

%: %.cpp
	$(CC) $(CFLAGS) $(LFLAGS) $(INCLUDES) $< -o $@

.PHONY: clean
clean:
	rm -f $(BIN)

