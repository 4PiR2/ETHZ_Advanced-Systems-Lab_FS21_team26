SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin

EXE := $(BIN_DIR)/main
SRC := $(wildcard $(SRC_DIR)/*.c)
SRC += $(wildcard $(SRC_DIR)/*.cpp)

OBJ := $(SRC:$(SRC_DIR)/%=$(OBJ_DIR)/%.o)

CC := g++
# modify #define branches here
# DEF := -Dxxx

# CPPFLAGS := -MMD -MP $(DEF) -O3 -march=native -ffast-math
CPPFLAGS := -MMD -MP $(DEF) -O3 -march=native -ffast-math -fno-tree-vectorize
# To compile AVX-512 version, set -march=icelake-client flag!
CFLAGS   := -Wall
LDFLAGS  := -Llib
LDLIBS   := -lm

.PHONY: all clean run

all: run

run: compile
	$(BIN_DIR)/main

compile: $(BIN_DIR)/main

$(BIN_DIR)/main: src/main.cpp | $(BIN_DIR)
	$(CC) $(CPPFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BIN_DIR):
	@mkdir -p $@

clean:
	@rm -rf $(BIN_DIR) 2>/dev/null

# make clean && make run

