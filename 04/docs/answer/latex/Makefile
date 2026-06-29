.PHONY: all pdf check-source refresh-source-manifest clean

TEX_FILES := answer.tex

all: check-source pdf

check-source:
	python3 check_source.py

refresh-source-manifest:
	python3 check_source.py --write

pdf:
	latexmk -xelatex -interaction=nonstopmode -halt-on-error $(TEX_FILES)

clean:
	latexmk -C $(TEX_FILES)
