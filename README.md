# CA-MHD — Conditional Calibration Contracts under Coupled Load

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23540789.svg)](https://doi.org/10.5281/zenodo.23540789)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-jsantillana%2Fca--mhd-FFD21E.svg?logo=huggingface&labelColor=ff5a5f)](https://huggingface.co/jsantillana/ca-mhd-weights)

Reproducible code and ACM paper for CA-MHD.

## Build paper
```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Reproduce experiments
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python scripts/make_report.py
```
One-click report regenerates tables and figures 1-4.

Seeds:
- train = seed, val = 100+seed, test = 200+seed, rollout = 300+seed

Results:
- `results/campaign/nets/` → per-cell networks
- `results/f5/` → transient windows
- `results/router_sweep/` → router equalization

Pre-registration and amendments are in `PRE-REGISTRATION.md` in the repo.

## License
CC-BY-4.0 for paper, Apache-2.0 for code.
