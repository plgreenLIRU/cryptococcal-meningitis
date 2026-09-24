# Modelling of Cryptococcal meningitis antifungal regimes

## Introduction

This repository contains the code and data used to generate the results in the manuscript titled _An Adaptive Approach for the Evaluation of Novel Antifungal Regimens for Cryptococcal Meningitis_.

## Installation

### Python

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

### R

This package uses [`renv`](https://rstudio.github.io/renv/). Make sure it is installed to your system R,
then in an R console run:

```R
renv::restore()
```

## Prerequisites

This repository requires raw data from AMBITION, which needs to be placed in `data/Arm2.csv`.
Because the dataset contains individual-level sensitive data, it is not shared here.
Please contact the authors to gain access to the raw data.

## Running

Run the scripts in the following sequence:

```bash
# generate population posterior samples
uv run Bayes.py

# predict new patient trajectories
uv run Predict.py

# plot new posterior trajectory overlayed onto population posteriors (Figure 5 in manuscript)
Rscript posterior_estimates.R
```

## Tests

To run the tests, execute the following command:

```bash
uv run pytest
```

## Authors

Peter L Green <plgreen@liverpool.ac.uk>

Alessandro Gerada <alessandro.gerada2@liverpool.ac.uk>

William Hope <hopew@liverpool.ac.uk>
