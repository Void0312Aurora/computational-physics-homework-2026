.PHONY: all run docs problem1 problem2 problem3 problem4 problem5 clean

all: docs

run:
	python3 scripts/hw07_integrals.py

docs: run
	python3 docs/answer/render_docs.py

problem1:
	python3 scripts/problem1.py

problem2:
	python3 scripts/problem2.py

problem3:
	python3 scripts/problem3.py

problem4:
	python3 scripts/problem4.py

problem5:
	python3 scripts/problem5.py

clean:
	rm -f result/*.csv result/*.png result/*.log
	rm -f docs/answer/answer.docx docs/answer/answer.pdf
