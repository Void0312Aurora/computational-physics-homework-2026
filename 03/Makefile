CC := gcc
CFLAGS := -std=gnu11 -O2 -Wall -Wextra -fexcess-precision=standard
LDLIBS := -lm -lquadmath
TARGET := hw03
VENV_PYTHON := /home/void0312/Cpy/.venv/bin/python
ANALYSIS := plot_results.py
DOC_RENDER := docs/answer/render_docs.py

.PHONY: all run analyze docs problem1 problem2 problem3 clean

all: $(TARGET)

$(TARGET): solution.c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

run: $(TARGET)
	mkdir -p result
	./$(TARGET) > result/temp-01.log

problem1:
	mkdir -p result
	$(CC) $(CFLAGS) scripts/problem1.c -o hw03_problem1 $(LDLIBS)
	./hw03_problem1

problem2:
	mkdir -p result
	$(CC) $(CFLAGS) scripts/problem2.c -o hw03_problem2 $(LDLIBS)
	./hw03_problem2

problem3:
	mkdir -p result
	$(CC) $(CFLAGS) scripts/problem3.c -o hw03_problem3 $(LDLIBS)
	./hw03_problem3

analyze: run
	$(VENV_PYTHON) $(ANALYSIS)

docs: analyze
	$(VENV_PYTHON) $(DOC_RENDER)

clean:
	rm -f $(TARGET) hw03_problem1 hw03_problem2 hw03_problem3
	rm -rf result
	rm -f docs/answer/answer.docx docs/answer/answer.pdf docs/answer/answer.doc
