.PHONY: all pdf clean

all: pdf

pdf:
	latexmk -xelatex -interaction=nonstopmode -halt-on-error answer.tex

clean:
	latexmk -C answer.tex
