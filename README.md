# ICLR 2026 — AI Safety Paper Analysis

Classifies all 5,352 accepted ICLR 2026 papers into four categories (Ethics & Fairness, Truthfulness & Reliability, General Capabilities, AI Safety) and breaks the safety papers down into 17 subdomains using DeepSeek V4 Flash with reasoning.

## Pipeline

1. [fetch_papers.py](fetch_papers.py) — pull all ICLR 2026 submissions from the OpenReview API → `iclr2026_papers.csv`
2. [classify.py](classify.py) — classify each paper with async parallel inference (50 workers) → `safety_results.csv`
3. [visualize.py](visualize.py) — generate plots and filtered CSVs

Classification prompt is in [prompt.txt](prompt.txt). Previous runs are archived in `runs/`.

## Results

**412 out of 5,352 papers (7.7%) classified as AI Safety.**

![AI Safety at ICLR 2026](classification_overview.png)

### Safety score distribution

![Score Distribution](plot_score_distribution.png)

### Output files

| File | Description |
|------|-------------|
| [safety_results.csv](safety_results.csv) | Full classification results for all 5,352 papers |
| [safety_papers_only.csv](safety_papers_only.csv) | 412 safety papers with subarea, subdomain, and score |
| [ethics_fairness_papers.csv](ethics_fairness_papers.csv) | 156 ethics & fairness papers |
| [truthfulness_reliability_papers.csv](truthfulness_reliability_papers.csv) | 474 truthfulness & reliability papers |
