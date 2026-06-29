CC := gcc
CFLAGS := -std=gnu11 -O2 -Wall -Wextra -fexcess-precision=standard
LDLIBS := -lm -lquadmath
VENV_PYTHON := /home/void0312/Cpy/.venv/bin/python
ANALYSIS := plot_results.py
DOC_RENDER := docs/answer/render_docs.py
LATEX_DIR := docs/answer/latex
LATEX_ANSWER_PDF := $(LATEX_DIR)/answer.pdf
SUBMISSION_SCRIPT_DIR := $(filter scripts,$(notdir $(CURDIR)))
RESULT_DIR := $(if $(SUBMISSION_SCRIPT_DIR),../results,result)
SOURCE_DIR := $(if $(wildcard scripts/problem1.c),scripts,.)
RESULT_DEFINE := -DHW03_RESULT_DIR=\"$(RESULT_DIR)\"
SUBMISSION_NAME := HW04.姜玥晟
SUBMISSION_DIR := $(SUBMISSION_NAME)
SUBMISSION_TAR := $(SUBMISSION_NAME).tar
SUBMISSION_DOCS := $(SUBMISSION_DIR)/docs
SUBMISSION_SCRIPTS := $(SUBMISSION_DIR)/scripts
SUBMISSION_RESULTS := $(SUBMISSION_DIR)/results

.PHONY: all run analyze docs latex-docs package submit problem1 problem2 problem3 clean

all: run

run:
	mkdir -p $(RESULT_DIR)
	$(CC) $(CFLAGS) $(RESULT_DEFINE) $(SOURCE_DIR)/problem1.c -o hw03_problem1 $(LDLIBS)
	./hw03_problem1 > $(RESULT_DIR)/temp-01.log
	$(CC) $(CFLAGS) $(RESULT_DEFINE) $(SOURCE_DIR)/problem2.c -o hw03_problem2 $(LDLIBS)
	./hw03_problem2 >> $(RESULT_DIR)/temp-01.log
	$(CC) $(CFLAGS) $(RESULT_DEFINE) $(SOURCE_DIR)/problem3.c -o hw03_problem3 $(LDLIBS)
	./hw03_problem3 >> $(RESULT_DIR)/temp-01.log

problem1:
	mkdir -p $(RESULT_DIR)
	$(CC) $(CFLAGS) $(RESULT_DEFINE) $(SOURCE_DIR)/problem1.c -o hw03_problem1 $(LDLIBS)
	./hw03_problem1

problem2:
	mkdir -p $(RESULT_DIR)
	$(CC) $(CFLAGS) $(RESULT_DEFINE) $(SOURCE_DIR)/problem2.c -o hw03_problem2 $(LDLIBS)
	./hw03_problem2

problem3:
	mkdir -p $(RESULT_DIR)
	$(CC) $(CFLAGS) $(RESULT_DEFINE) $(SOURCE_DIR)/problem3.c -o hw03_problem3 $(LDLIBS)
	./hw03_problem3

analyze: run
	$(VENV_PYTHON) $(ANALYSIS)

docs: analyze
	$(VENV_PYTHON) $(DOC_RENDER)

latex-docs:
	$(MAKE) -C $(LATEX_DIR)

package: docs latex-docs
	rm -rf "$(SUBMISSION_DIR)"
	mkdir -p "$(SUBMISSION_DOCS)" "$(SUBMISSION_SCRIPTS)" "$(SUBMISSION_RESULTS)"
	install -m 0644 $(LATEX_ANSWER_PDF) "$(SUBMISSION_DOCS)/answer.pdf"
	install -m 0644 Makefile plot_results.py "$(SUBMISSION_SCRIPTS)"
	install -m 0644 $(SOURCE_DIR)/problem1.c "$(SUBMISSION_SCRIPTS)/problem1.c"
	install -m 0644 $(SOURCE_DIR)/problem2.c "$(SUBMISSION_SCRIPTS)/problem2.c"
	install -m 0644 $(SOURCE_DIR)/problem3.c "$(SUBMISSION_SCRIPTS)/problem3.c"
	install -m 0644 $(RESULT_DIR)/* "$(SUBMISSION_RESULTS)"
	tar -cf "$(SUBMISSION_TAR)" "$(SUBMISSION_DIR)"

submit: package

clean:
	rm -f hw03 hw03_problem1 hw03_problem2 hw03_problem3
	rm -rf $(RESULT_DIR)
	rm -f docs/answer/answer.docx docs/answer/answer.pdf docs/answer/answer.doc
