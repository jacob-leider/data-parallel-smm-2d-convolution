PY := python3.11 -m
PIP := pip
C_FORMAT := clang-format
PY_FORMAT := autopep8

EXAMPLE_DIR := examples
TEST_DIR := tests
C_FILES := $(shell find . -type f \( -name '*.c' -o -name '*.cpp' \) -not -path "./venv/*")
H_FILES := $(shell find . -type f \( -name '*.h' -o -name '*.hpp' \) -not -path "./venv/*")
PY_FILES := $(shell find . -name '*.py' -not -path "./venv/*")

clangd:
	$(PY) gen_clangd

extension:
	$(PIP) install .

setup: clangd extension

run_%:
	$(PY) $(EXAMPLE_DIR).$*

# TODO make a test for doing all tests
test_%:
	$(PY) $(TEST_DIR).$*

test:
	$(PY) $(TEST_FILE)

format: $(C_FILES) $(H_FILES) $(PY_FILES)
	$(C_FORMAT) -i $(C_FILES) $(H_FILES)
	$(PY_FORMAT) --in-place $(PY_FILES)
