VENV     := .venv
PY       := $(VENV)/bin/python
PIP      := $(VENV)/bin/pip
HF_ENV   := HF_HOME=.hf-cache
VIDEO    ?= video.mp4
MODEL    ?= medium
ARGS     ?=

.PHONY: venv install run serve check clean

venv:
	python3 -m venv $(VENV)
	$(PIP) install -U pip

install: venv
	$(PIP) install -r requirements.txt

run: install
	$(HF_ENV) $(PY) app.py $(VIDEO) -m $(MODEL) $(ARGS)

serve: install
	$(HF_ENV) $(VENV)/bin/streamlit run ui.py

# SRT writer sanity checks (no video needed)
check: install
	$(PY) check.py

clean:
	rm -rf $(VENV) __pycache__ .hf-cache
