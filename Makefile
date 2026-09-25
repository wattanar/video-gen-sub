VENV     := .venv
PY       := $(VENV)/bin/python
PIP      := $(VENV)/bin/pip
VIDEO    ?= video.mp4
MODEL    ?= medium
ARGS     ?=

.PHONY: venv install whisper run serve check clean

venv:
	python3 -m venv $(VENV)
	$(PIP) install -U pip

install: venv
	$(PIP) install -r requirements.txt

# Build whisper.cpp with the AMD Vulkan backend (one-time, ~few minutes)
whisper:
	@if [ ! -x .whisper-cpp/build/bin/whisper-cli ]; then \
		git clone --depth 1 https://github.com/ggml-org/whisper.cpp .whisper-cpp && \
		cmake -B .whisper-cpp/build -S .whisper-cpp -DGGML_VULKAN=ON && \
		cmake --build .whisper-cpp/build -j$(nproc); fi

run: install whisper
	$(PY) app.py $(VIDEO) -m $(MODEL) --device vulkan $(ARGS)

serve: install whisper
	$(VENV)/bin/streamlit run ui.py

# SRT writer sanity checks (no video needed)
check: install
	$(PY) check.py

clean:
	rm -rf $(VENV) __pycache__ .hf-cache .whisper-cpp models
