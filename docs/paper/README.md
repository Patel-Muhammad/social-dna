# Research paper (IEEE conference format)

| File | What it is |
|---|---|
| `social_dna_paper.tex` | LaTeX source (IEEEtran, two-column) |
| `social_dna_paper.pdf` | Compiled paper |
| `figures/*.pdf` | The 4 figures, generated from the real database |

## Edit and recompile the PDF

Edit `social_dna_paper.tex` in any editor (Overleaf, TeXstudio, VS Code + LaTeX Workshop), then compile it:

```bat
cd docs\paper
tectonic social_dna_paper.tex
```

`tectonic` (https://tectonic-typesetting.github.io, one download, fetches packages automatically) is the easiest option on Windows. A normal
TeX Live or MiKTeX install also works: `pdflatex social_dna_paper.tex` (run twice). Upload the whole `docs/paper` folder to Overleaf to
compile online. The template falls back to the Times-style fonts of IEEEtran when compiled with pdfLaTeX.

Fill in the placeholders on lines 22-25 and 256 of `social_dna_paper.tex` (`[Author Name]`, `[Department, University]`, `[City, Country]`, `[email address]`, `[repository URL]`). The exact table and LaTeX snippet are in the main `README.md`, section "Research paper".

## Regenerate the numbers and figures from the data

All numbers in the paper come from the SQLite database and `data/evaluation/results.json`. After re-running the analysis or the evaluation:

```bat
pip install -r requirements-dev.txt
python scripts\build_paper.py
```

This rebuilds the figures and **overwrites `social_dna_paper.tex` from `scripts/paper_template.tex`**. If you want to keep your manual
edits to the text, either edit `scripts\paper_template.tex` instead (placeholders such as `@@Q@@` are filled automatically), or do not
run the script again.

## Before submitting

- Check every reference against the original source (they were written from memory) and add DOIs if your venue requires them.
- Replace the author placeholders (see above) and check the venue's page limit and template version.
