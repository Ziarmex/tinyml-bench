# SPDX-License-Identifier: MIT
PYTHON ?= python3

.PHONY: test report train clean install

test:
	$(PYTHON) -m pytest tests -q

report:
	$(PYTHON) tools/gen_report.py

train:
	PYTHONPATH=src $(PYTHON) -m tinyml_bench train

install:
	$(PYTHON) -m pip install -e .

clean:
	rm -f results/*.tflite results/*.h5
	rm -rf .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
