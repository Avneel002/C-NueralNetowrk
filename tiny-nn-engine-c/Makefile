CC = gcc
CFLAGS = -Wall -Wextra -Werror -O2 -Iinclude -std=c99
LDFLAGS = -lm
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = .
TEST_DIR = tests
BENCH_DIR = benchmarks
SOURCES = $(wildcard $(SRC_DIR)/*.c)
OBJECTS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SOURCES))
TARGET = $(BIN_DIR)/tiny_nn
TEST_SOURCES = $(wildcard $(TEST_DIR)/*.c)
TEST_TARGETS = $(patsubst $(TEST_DIR)/%.c,$(BIN_DIR)/%,$(TEST_SOURCES))
all: $(TARGET)
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $(TARGET)
test: $(TEST_TARGETS)
$(BIN_DIR)/test_%: $(TEST_DIR)/test_%.c $(filter-out $(OBJ_DIR)/main.o,$(OBJECTS))
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@
check: test
	@for test in $(TEST_TARGETS); do \
		echo "Running $$test..."; \
		./$$test || exit 1; \
	done
valgrind: $(TARGET)
	valgrind --leak-check=full --show-leak-kinds=all ./$(TARGET)
clean:
	rm -rf $(OBJ_DIR) $(TARGET) $(TEST_TARGETS)

.PHONY: all test check clean valgrind