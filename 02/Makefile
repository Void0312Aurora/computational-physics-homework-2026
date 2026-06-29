CC := gcc
CFLAGS := -std=gnu11 -O2 -Wall -Wextra
LDLIBS := -lm -lquadmath
VENV_PYTHON := /home/void0312/Cpy/.venv/bin/python
DOC_RENDER := docs/answer/render_docs.py
RESULT_DIR := result
SOURCE_DIR := $(if $(wildcard scripts/problem1.c),scripts,.)
PROBLEM_TARGETS := hw02_problem1 hw02_problem2 hw02_problem3 hw02_problem4 hw02_problem5
SUBMISSION_NAME := HW03.姜玥晟
SUBMISSION_DIR := $(SUBMISSION_NAME)
SUBMISSION_TAR := $(SUBMISSION_NAME).tar
SUBMISSION_DOCS := $(SUBMISSION_DIR)/docs
SUBMISSION_SCRIPTS := $(SUBMISSION_DIR)/scripts
SUBMISSION_RESULTS := $(SUBMISSION_DIR)/results

.PHONY: all run emit-run-log result-log docs package submit sync-results check-project-root problem1 problem2 problem3 problem4 problem5 clean

all: $(PROBLEM_TARGETS)

hw02_problem1: $(SOURCE_DIR)/problem1.c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

hw02_problem2: $(SOURCE_DIR)/problem2.c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

hw02_problem3: $(SOURCE_DIR)/problem3.c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

hw02_problem4: $(SOURCE_DIR)/problem4.c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

hw02_problem5: $(SOURCE_DIR)/problem5.c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

run: $(PROBLEM_TARGETS)
	$(MAKE) --no-print-directory emit-run-log

emit-run-log:
	@printf 'HW/02 numerical experiments\n'
	@printf '===========================\n\n'
	@printf 'Run metadata\n'
	@printf 'compiler version              : '
	@$(CC) --version | sed -n '1p'
	@printf 'compiler flags                : $(CFLAGS)\n'
	@printf 'sizeof(float/double/f128)     : 4 / 8 / 16 bytes\n'
	@printf 'Gregory terms                 : 500000\n'
	@printf 'Machin tolerance              : 1.0e-35\n'
	@printf 'recurrence n max              : 40\n\n'
	@./hw02_problem1
	@printf '\n'
	@./hw02_problem2
	@printf '\n'
	@./hw02_problem3
	@printf '\n'
	@./hw02_problem4
	@printf '\n'
	@./hw02_problem5
	@printf '\n'

problem1: hw02_problem1
	./hw02_problem1

problem2: hw02_problem2
	./hw02_problem2

problem3: hw02_problem3
	./hw02_problem3

problem4: hw02_problem4
	./hw02_problem4

problem5: hw02_problem5
	./hw02_problem5

check-project-root:
	@test -f "$(DOC_RENDER)" || { echo "This target is only available from the HW/02 project root."; exit 2; }

result-log: $(PROBLEM_TARGETS)
	mkdir -p $(RESULT_DIR)
	$(MAKE) --no-print-directory emit-run-log > $(RESULT_DIR)/temp-01.log

docs: check-project-root result-log
	$(VENV_PYTHON) $(DOC_RENDER)

sync-results: docs
	mkdir -p "$(SUBMISSION_RESULTS)"
	install -m 0644 "$(RESULT_DIR)/temp-01.log" "$(SUBMISSION_RESULTS)/temp-01.log"

package: docs
	rm -rf "$(SUBMISSION_DIR)"
	mkdir -p "$(SUBMISSION_DOCS)" "$(SUBMISSION_SCRIPTS)" "$(SUBMISSION_RESULTS)"
	install -m 0644 docs/answer/answer.pdf "$(SUBMISSION_DOCS)/answer.pdf"
	install -m 0644 Makefile "$(SUBMISSION_SCRIPTS)"
	install -m 0644 $(SOURCE_DIR)/problem1.c "$(SUBMISSION_SCRIPTS)/problem1.c"
	install -m 0644 $(SOURCE_DIR)/problem2.c "$(SUBMISSION_SCRIPTS)/problem2.c"
	install -m 0644 $(SOURCE_DIR)/problem3.c "$(SUBMISSION_SCRIPTS)/problem3.c"
	install -m 0644 $(SOURCE_DIR)/problem4.c "$(SUBMISSION_SCRIPTS)/problem4.c"
	install -m 0644 $(SOURCE_DIR)/problem5.c "$(SUBMISSION_SCRIPTS)/problem5.c"
	install -m 0644 "$(RESULT_DIR)/temp-01.log" "$(SUBMISSION_RESULTS)/temp-01.log"
	tar -cf "$(SUBMISSION_TAR)" "$(SUBMISSION_DIR)"

submit: package

clean:
	rm -f $(PROBLEM_TARGETS)
