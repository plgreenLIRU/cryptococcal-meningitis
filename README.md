# Modelling of Cryptococcal meningitis antifungal regimes

## Introduction

This repository contains the code and data used to generate the results in the manuscript titled _An Adaptive Approach for the Evaluation of Novel Antifungal Regimens for Cryptococcal Meningitis_.

## Installation

This package uses [`uv`](https://docs.astral.sh/uv/). Once you have installed `uv`, run:

```bash
# install dependencies from pyproject.toml
uv sync

# run the main script
uv run main.py

# generate plots
uv run Predict.py
```

If you need to add the `samplers` dependency from GitHub:

```bash
uv add "samplers @ git+https://github.com/plgreenLIRU/samplers.git"
```
