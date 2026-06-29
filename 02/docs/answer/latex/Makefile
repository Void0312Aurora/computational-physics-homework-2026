.PHONY: all pdf clean

TEX_FILES := answer.tex

all: pdf

pdf:
	latexmk -xelatex -interaction=nonstopmode -halt-on-error $(TEX_FILES)

clean:
	latexmk -C $(TEX_FILES)
