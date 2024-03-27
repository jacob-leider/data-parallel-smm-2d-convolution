PY := python3.11 -m
PIP := pip
C_FORMAT := clang-format
PY_FORMAT := autopep8

TEST_DIR := tests
UNIT_TEST_DIR := $(TEST_DIR).unit
INTEGRATION_TEST_DIR := $(TEST_DIR).integration
C_FILES := $(shell find . -name '*.cpp' -not -path "./venv/*")
PY_FILES := $(shell find . -name '*.py' -not -path "./venv/*")

clangd:
	$(PY) gen_clangd

ext:
	$(PIP) install .

ext_e:
	$(PIP) install --editable .

setup: clangd extension

test:
	$(PY) $(TEST_DIR)

utest:
	$(PY) $(UNIT_TEST_DIR)

utest.%:
	$(PY) $(UNIT_TEST_DIR).$*

itest:
	$(PY) $(INTEGRATION_TEST_DIR)

itest.%:
	$(PY) $(INTEGRATION_TEST_DIR).$*

format: $(C_FILES) $(PY_FILES)
	$(C_FORMAT) -i $(C_FILES)
	$(PY_FORMAT) --in-place $(PY_FILES)
