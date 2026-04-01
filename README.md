# CivicTriage-AI

## How I Built a Preference-Aligned Routing PoC for Municipal-Style Service Requests

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![Title architecture](https://raw.githubusercontent.com/aniket-work/CivicTriage-AI/5b9f2d1fd1cb81b527cecb87622e84aaa8560d5f/images/title_diagram.png)

CivicTriage-AI is an experimental, solo-built proof of concept that routes synthetic “311-style” citizen requests to a small set of municipal-style departments. The repository demonstrates a practical training story that mirrors how I think about modern agent development: start with a supervised policy, then nudge it with reviewer preferences so the behavior matches operational nuance that raw labels alone rarely capture.

This is not production software. It is a compact Python workspace meant to show how I structure data generation, evaluation, and simple alignment steps without relying on a proprietary cluster or a full-scale foundation-model fine-tuning run.

## What Problem This PoC Tries to Approximate

Public service desks often receive messy natural language. A single sentence might mention water pressure, noise, and a sidewalk crack. Routing rules written by hand drift out of date, while end-to-end LLM prompts can be brittle when departments disagree on edge cases. In my experiments, the interesting middle ground is a lightweight classifier trained on representative text, then refined with pairwise feedback (“prefer route A over route B for this phrasing”) that stands in for human review.

The code here does not call an external LLM API. Instead, it uses a transparent bag-of-words representation and a multinomial logistic regression head so the entire pipeline runs on a laptop and remains easy to audit.

## Architecture Overview

![High-level architecture](https://raw.githubusercontent.com/aniket-work/CivicTriage-AI/5b9f2d1fd1cb81b527cecb87622e84aaa8560d5f/images/architecture_diagram.png)

The pipeline has four conceptual stages:

1. Synthetic corpus generation for controlled experiments.
2. Supervised fine-tuning (SFT) of a multinomial logistic policy on labeled text.
3. A preference-alignment augmentation that duplicates reviewer-approved routes and lightly annotates contrastive notes.
4. Offline evaluation with accuracy and macro-averaged F1, plus simple charts for quick visual comparison.

## Sequence View

![Request handling sequence](https://raw.githubusercontent.com/aniket-work/CivicTriage-AI/5b9f2d1fd1cb81b527cecb87622e84aaa8560d5f/images/sequence_diagram.png)

## Training Flow

![Training flow](https://raw.githubusercontent.com/aniket-work/CivicTriage-AI/5b9f2d1fd1cb81b527cecb87622e84aaa8560d5f/images/flow_diagram.png)

## Cover Animation

![Demo animation](https://raw.githubusercontent.com/aniket-work/CivicTriage-AI/5b9f2d1fd1cb81b527cecb87622e84aaa8560d5f/images/title-animation.gif)

The animated asset summarizes the narrative: run the CLI, inspect the ASCII summary table, then review the evaluation snapshot panel.

## Repository Layout

```
CivicTriage-AI/
├── LICENSE
├── README.md
├── main.py                 # CLI entrypoint
├── requirements.txt
├── images/                 # Diagrams, GIF, and metric plots for documentation
├── output/                 # Generated charts when you run the CLI (gitignored PNGs)
└── src/civic_triage/
    ├── labels.py           # Department label definitions
    ├── synthetic.py        # Synthetic requests + preference pairs
    ├── modeling.py         # TF-IDF + logistic regression + alignment helpers
    ├── plots.py            # Matplotlib charts written to disk
    └── reporting.py        # ASCII tables for terminal output
```

## How I Run It Locally

1. Create an isolated environment inside this folder:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Execute the training and evaluation pipeline:

```bash
python main.py --seed 42 --n-per-class 120 --pref-noise 0.22
```

3. Inspect the printed ASCII tables in the terminal and the generated charts under `output/`. The repository already copies representative charts into `images/` for documentation consistency.

### CLI Parameters

| Flag | Meaning |
| --- | --- |
| `--seed` | Controls reproducibility for synthetic data and model fitting. |
| `--n-per-class` | Number of synthetic examples per department label. |
| `--pref-noise` | Fraction of training rows that generate synthetic preference pairs. |

## Metrics and Visual Outputs

Running `main.py` produces:

1. `output/label_distribution.png` — bar chart of class counts in the training split.
2. `output/metrics_compare.png` — side-by-side accuracy and macro F1 for the supervised model versus the preference-augmented refit.

Representative copies used in this README live under `images/label_distribution.png` and `images/metrics_compare.png`.

## Design Choices I Made on Purpose

1. **Transparency over scale**: A TF-IDF + logistic regression stack is easy to inspect. In my opinion, that matters when explaining routing decisions to stakeholders who do not live inside deep learning frameworks daily.
2. **Preference pairs without a full DPO implementation**: I approximate reviewer intent by duplicating chosen labels and adding sparse contrastive text hints. This keeps the PoC small while still communicating the alignment idea.
3. **Synthetic data only**: Real municipal text requires careful handling of privacy and retention policies. I keep this repository strictly synthetic so it can remain public without exposing anyone’s personal information.

## Limitations I Am Up Front About

1. The vocabulary is templated, so metrics can look optimistic compared to messy real-world corpora.
2. The alignment step is a stand-in for rigorous offline RL or preference optimization at scale.
3. There is no online feedback loop, A/B test harness, or integration with a ticketing system.

## Ethics and Responsible Use

Even as a toy dataset, civic routing touches sensitive domains. I designed the examples to avoid personal data, and I recommend treating any real deployment path as a separate effort with formal governance, accessibility review, and human oversight for escalations.

## License

This project is released under the MIT License. See `LICENSE`.

## How to Cite or Fork

If this repository helps your own experiments, feel free to fork it. Please keep the disclaimer in mind: this is a personal research PoC, not an endorsement of any vendor stack and not a blueprint for unattended automation in public services.
