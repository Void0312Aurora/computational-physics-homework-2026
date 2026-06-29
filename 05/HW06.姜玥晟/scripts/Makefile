PYTHON ?= python3

SUBMISSION_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
SCRIPT_DIR := $(SUBMISSION_ROOT)/scripts
export PYTHONPATH := $(SCRIPT_DIR)$(if $(PYTHONPATH),:$(PYTHONPATH))

.PHONY: problem1 problem2 problem3 problem4_project2 project2_manifest reproduce_project2_report_manifest

problem1:
	cd $(SUBMISSION_ROOT) && $(PYTHON) scripts/problem1.py

problem2:
	cd $(SUBMISSION_ROOT) && $(PYTHON) scripts/problem2.py

problem3:
	cd $(SUBMISSION_ROOT) && $(PYTHON) scripts/problem3.py

problem4_project2:
	cd $(SUBMISSION_ROOT) && $(PYTHON) scripts/problem4_project2.py

project2_manifest:
	cd $(SUBMISSION_ROOT) && $(PYTHON) -m project2_pi.manifest_pi_output --base .

reproduce_project2_report_manifest: project2_manifest
