SHELL := bash
PYTHON ?= python

.PHONY: run docs problem1 problem2 problem3 problem4

run:
	mkdir -p result
	set -o pipefail; $(PYTHON) solution.py | tee result/temp-01.log

docs: run
	set -o pipefail; $(PYTHON) docs/answer/render_docs.py | tee result/temp-02.log

problem1:
	$(PYTHON) scripts/problem1.py

problem2:
	$(PYTHON) scripts/problem2.py

problem3:
	$(PYTHON) scripts/problem3.py

problem4:
	$(PYTHON) scripts/problem4.py
