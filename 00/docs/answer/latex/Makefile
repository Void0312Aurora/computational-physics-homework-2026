.PHONY: all pdf clean

TEX := answer.tex

all: pdf

pdf:
	latexmk -xelatex -interaction=nonstopmode -halt-on-error $(TEX)

clean:
	latexmk -C $(TEX)
