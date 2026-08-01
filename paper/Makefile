# Latent Teleportation paper.
#
#   make          regenerate tables from raw results, then build the PDF
#   make tables   regenerate generated/*.tex only
#   make clean    remove build artifacts (keeps the PDF)
#
# The tables and the numeric macros used in the prose are generated from the raw
# result JSON by scripts/paper_analysis.py, so no figure in the text can drift
# away from the measurement it came from.

PAPER  := latent_teleportation
PYTHON ?= python3

.PHONY: all tables clean distclean

all: $(PAPER).pdf

tables:
	cd .. && $(PYTHON) scripts/paper_analysis.py --out paper/generated

$(PAPER).pdf: $(PAPER).tex refs.bib tables
	latexmk -pdf -halt-on-error -interaction=nonstopmode $(PAPER).tex

clean:
	latexmk -c $(PAPER).tex 2>/dev/null || true
	rm -f $(PAPER).bbl $(PAPER).run.xml

distclean: clean
	rm -f $(PAPER).pdf
	rm -rf generated
