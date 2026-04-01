PYTHON ?= /home/lee/code/stock/.venv/bin/python
REPO_ROOT := $(abspath $(CURDIR)/../..)
CHRONOS_SRC ?= $(abspath $(REPO_ROOT)/../stock/chronos-forecasting/src)
BUILD_DIR := $(CURDIR)/build
INCLUDE_DIR := $(CURDIR)/include
LIB_PATH := $(BUILD_DIR)/libcutechronos_bridge.so
C_RUNNER := $(BUILD_DIR)/chronos_c_runner
GO_RUNNER := $(BUILD_DIR)/chronos_go_runner

PY_INCLUDE := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('INCLUDEPY'))")
PY_LIBDIR := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
PY_LDVERSION := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('LDVERSION') or sysconfig.get_config_var('VERSION'))")
PY_PROGRAM := $(shell $(PYTHON) -c "import sys; print(sys.executable)")
PY_HOME := $(shell $(PYTHON) -c "import sys; print(sys.base_prefix)")
DEFAULT_PYTHONPATH := $(REPO_ROOT):$(CHRONOS_SRC)

CFLAGS := -O2 -fPIC -Wall -Wextra -I$(INCLUDE_DIR) -I$(PY_INCLUDE)
CPPFLAGS := -DCUTECHRONOS_DEFAULT_PROGRAM=\"$(PY_PROGRAM)\" \
	-DCUTECHRONOS_DEFAULT_PYTHONHOME=\"$(PY_HOME)\" \
	-DCUTECHRONOS_DEFAULT_PYTHONPATH=\"$(DEFAULT_PYTHONPATH)\"
LDFLAGS_SO := -shared -L$(PY_LIBDIR) -lpython$(PY_LDVERSION) -Wl,-rpath,$(PY_LIBDIR)
LDFLAGS_BIN := -L$(BUILD_DIR) -lcutechronos_bridge -Wl,-rpath,\$$ORIGIN

.PHONY: all clean

all: $(LIB_PATH) $(C_RUNNER) $(GO_RUNNER)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(LIB_PATH): $(BUILD_DIR) src/cutechronos_bridge.c include/cutechronos_bridge.h
	gcc $(CFLAGS) $(CPPFLAGS) src/cutechronos_bridge.c -o $@ $(LDFLAGS_SO)

$(C_RUNNER): $(BUILD_DIR) c/main.c $(LIB_PATH) include/cutechronos_bridge.h
	gcc -O2 -Wall -Wextra -I$(INCLUDE_DIR) c/main.c -o $@ $(LDFLAGS_BIN)

$(GO_RUNNER): $(LIB_PATH) go/main.go go/go.mod include/cutechronos_bridge.h
	cd go && CGO_ENABLED=1 go build -o ../build/chronos_go_runner .

clean:
	rm -rf $(BUILD_DIR)
