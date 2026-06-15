CC := gcc
CFLAGS := -std=gnu11 -O2 -Wall -Wextra
LDLIBS := -lm -lquadmath
TARGET := hw02
VENV_PYTHON := /home/void0312/Cpy/.venv/bin/python
DOC_RENDER := docs/answer/render_docs.py

.PHONY: all run docs problem1 problem2 problem3 problem4 problem5 clean

all: $(TARGET)

$(TARGET): solution.c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

run: $(TARGET)
	./$(TARGET)

problem1:
	$(CC) $(CFLAGS) scripts/problem1.c -o hw02_problem1 $(LDLIBS)
	./hw02_problem1

problem2:
	$(CC) $(CFLAGS) scripts/problem2.c -o hw02_problem2 $(LDLIBS)
	./hw02_problem2

problem3:
	$(CC) $(CFLAGS) scripts/problem3.c -o hw02_problem3 $(LDLIBS)
	./hw02_problem3

problem4:
	$(CC) $(CFLAGS) scripts/problem4.c -o hw02_problem4 $(LDLIBS)
	./hw02_problem4

problem5:
	$(CC) $(CFLAGS) scripts/problem5.c -o hw02_problem5 $(LDLIBS)
	./hw02_problem5

docs: $(TARGET)
	./$(TARGET) > result/temp-01.log
	$(VENV_PYTHON) $(DOC_RENDER)

clean:
	rm -f $(TARGET) hw02_problem1 hw02_problem2 hw02_problem3 hw02_problem4 hw02_problem5
