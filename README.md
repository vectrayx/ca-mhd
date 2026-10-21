# CA-MHD — Conditional Calibration Contracts under Coupled Load

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
