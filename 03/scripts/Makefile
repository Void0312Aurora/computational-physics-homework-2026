CC := gcc
CFLAGS := -std=gnu11 -O2 -Wall -Wextra -fexcess-precision=standard
LDLIBS := -lm -lquadmath
RESULT_DIR ?= ../results
RESULT_DEFINE := -DHW03_RESULT_DIR=\"$(RESULT_DIR)\"

.PHONY: all run problem1 problem2 problem3 clean

all: run

run:
	mkdir -p $(RESULT_DIR)
	$(CC) $(CFLAGS) $(RESULT_DEFINE) problem1.c -o hw03_problem1 $(LDLIBS)
	./hw03_problem1 > $(RESULT_DIR)/temp-01.log
	$(CC) $(CFLAGS) $(RESULT_DEFINE) problem2.c -o hw03_problem2 $(LDLIBS)
	./hw03_problem2 >> $(RESULT_DIR)/temp-01.log
	$(CC) $(CFLAGS) $(RESULT_DEFINE) problem3.c -o hw03_problem3 $(LDLIBS)
	./hw03_problem3 >> $(RESULT_DIR)/temp-01.log

problem1:
	mkdir -p $(RESULT_DIR)
	$(CC) $(CFLAGS) $(RESULT_DEFINE) problem1.c -o hw03_problem1 $(LDLIBS)
	./hw03_problem1

problem2:
	mkdir -p $(RESULT_DIR)
	$(CC) $(CFLAGS) $(RESULT_DEFINE) problem2.c -o hw03_problem2 $(LDLIBS)
	./hw03_problem2

problem3:
	mkdir -p $(RESULT_DIR)
	$(CC) $(CFLAGS) $(RESULT_DEFINE) problem3.c -o hw03_problem3 $(LDLIBS)
	./hw03_problem3

clean:
	rm -f hw03_problem1 hw03_problem2 hw03_problem3
