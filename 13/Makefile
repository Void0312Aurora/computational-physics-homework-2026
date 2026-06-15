.PHONY: all run render docs problem1 problem2 problem3 problem4 clean

all: docs

run:
	python3 scripts/hw13_ising.py

render:
	python3 docs/answer/render_docs.py

docs: run render

problem1:
	python3 scripts/problem1.py

problem2:
	python3 scripts/problem2.py

problem3:
	python3 scripts/problem3.py

problem4:
	python3 scripts/problem4.py

clean:
	rm -f result/*.csv result/*.png result/*.json result/*.log
	rm -f docs/answer/answer.docx docs/answer/answer.pdf
