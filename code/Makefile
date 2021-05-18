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

CPPFLAGS := -Iinclude -MMD -MP $(DEF) -O3 -march=native -ffast-math
# To compile AVX-512 version, set -march=icelake-client flag!
CFLAGS   := -Wall
LDFLAGS  := -Llib
LDLIBS   := -lm

.PHONY: all clean run

all: compile

run: compile
	$(EXE)

compile: $(EXE)

$(EXE): $(OBJ) | $(BIN_DIR)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(OBJ_DIR)/%.c.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%.cpp.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@ 
	# change the above line to g++ if wanted

$(BIN_DIR) $(OBJ_DIR):
	@mkdir -p $@

clean:
	@rm -rf $(BIN_DIR) $(OBJ_DIR) 2>/dev/null

test_benchmark: test/test_benchmark.c
	$(CC) $(CPPFLAGS) -o bin/test_benchmark test/test_benchmark.c && ./bin/test_benchmark

test_sym_aff: test/test_sym_aff.c
	$(CC) $(CPPFLAGS) -o bin/test_sym_aff test/test_sym_aff.c && ./bin/test_sym_aff

# -include $(OBJS:.o=.d)

