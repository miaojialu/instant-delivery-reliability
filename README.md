# Operational Coordination and Instant Food Delivery Reliability

Analysis source code accompanying the manuscript:

> Operational Coordination and Instant Food Delivery Reliability: Evidence from Two Chinese Platform-City Settings

## Overview

This study examines the associations between operational conditions and relative delivery delay in instant food delivery. It draws on two Chinese platform-city datasets and uses predictive modelling, SHAP explanations, accumulated local effects (ALE), interaction analysis, cross-context evaluation, and model-based scenario screening.

The analysis is observational. Model explanations and scenario comparisons do not establish causal effects or determine individual courier responsibility.

## What is shared

This repository shares analysis source code, a project description, and a dependency specification. The order-level datasets, including the original platform records and processed analysis tables, are not included.

Refer to the manuscript and supplementary material for the original data sources, sample construction, variable definitions, and analytical methods. Access to the underlying data is subject to the availability and terms of the respective sources.

## Repository structure

```text
.
├── README.md
├── requirements.txt
└── code/                 # Analysis scripts and/or notebooks
```

## Installation

The analysis uses Python. From the repository root, install the listed dependencies:

```bash
python -m pip install -r requirements.txt
```

The supplied dependency file lists unpinned core packages based on the methods described in the manuscript. It has not been verified against the analysis source code or the original execution environment. Additional packages and exact versions must be checked against the uploaded scripts before treating this specification as a complete environment for reproducing the results.

## Using the code

Running the code requires access to the corresponding input data. Before execution:

1. Obtain the required inputs from an authorized source and check that their fields and units match the requirements of the scripts.
2. Configure the input and output paths for your local environment.
3. Follow the execution order and input requirements documented in the scripts or notebooks.
4. Use the preprocessing rules, variable definitions, sample-split assignments or procedures, grouping keys, model settings, and random seeds corresponding to the reported analyses.

Geographic preprocessing also requires the relevant coordinate and spatial inputs. If route or POI queries are performed, users must configure their own service credentials where required. These inputs and credentials are not provided by this repository.

## Reproducibility

The code supports inspection of the analytical workflow. Reproducing the numerical results additionally requires the same input data, preprocessing, model specification, and a compatible software environment. Downloading this repository alone is insufficient to reproduce the reported results because the analysis datasets are not distributed here.

The full-feature models require all predictors used in the original analyses, including the geographic predictors. Omitting these variables changes the model specification and can change predictive performance, SHAP values, ALE profiles, interaction estimates, and scenario comparisons.

## Citation

Please cite the accompanying manuscript when using this code and acknowledge the original data sources as appropriate. Publication details and the article DOI can be added when available.
