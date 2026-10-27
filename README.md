# Operational Coordination and Instant Food Delivery Reliability

Processed data and analysis source code accompanying the manuscript:

> Operational Coordination and Instant Food Delivery Reliability: Evidence from Two Chinese Platform-City Settings

## Overview

The study examines how operational conditions at different stages of instant food delivery are associated with relative delivery delay. It uses two platform-city datasets and combines predictive modelling with SHAP explanations, accumulated local effects (ALE), interaction analysis, cross-context evaluation, and model-based scenario screening.

The analysis is observational. Model explanations and scenario comparisons describe fitted associations; they do not establish causal effects or determine individual courier responsibility.

## Public release

The public release consists of:

- Processed order-level data from the Beijing and Baoding datasets, with all geographic-information-related fields removed.
- Analysis source code.
- A dependency file for setting up the analysis environment.

The publicly released files are processed analysis data, not the complete original platform records. Refer to the manuscript and supplementary material for data sources, sample construction, variable definitions, and analytical methods.

## Geographic information excluded

All geographic-information-related fields are excluded from the public datasets. These exclusions cover merchant and customer coordinates, addresses and other location identifiers, spatial grid identifiers, route geometries or trajectories, and derived geographic variables used in the study, including route-based riding distance, POI density, and road-network measures.

Geographic inputs required to reconstruct these variables are not distributed with this release. Refer to the manuscript for the construction of the geographic variables used in the original analyses.

## Scope of reproducibility

Removing geographic variables limits what can be reproduced using the public data alone.

- Analyses whose required inputs are all retained can be rerun using the public datasets and the corresponding code.
- Coordinate processing, route queries, spatial aggregation, and construction of geographic variables require inputs excluded from this release.
- Full-model estimation, evaluation, explanations, cross-context analyses, and scenario screening that use excluded geographic predictors require those predictors. The public datasets alone are insufficient to reproduce these full-feature results.

The source code is provided to document the analytical workflow, including steps that may require excluded inputs. Such steps should be treated as requiring additional data, rather than as executable parts of a public-data-only workflow.

Training a model after removing geographic predictors changes its input specification. Its performance, SHAP values, ALE profiles, interactions, and scenario comparisons may therefore differ from those reported in the manuscript.

## Repository layout

The intended layout is:

```text
.
├── README.md
├── requirements.txt
├── data/                 # Public datasets with geographic fields removed
└── code/                 # Analysis scripts and/or notebooks
```

Data and source-code files should be placed in the corresponding folders when preparing the release. Their actual filenames should be used in the scripts. Field names, definitions, units, missing-value conventions, and any retained anonymous grouping or sample-split identifiers should be documented alongside the uploaded data.

## Environment and installation

The analysis uses Python. Install the listed packages from the repository root:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` currently lists unpinned core packages consistent with the methods described in the manuscript. It is a starting dependency specification, not a verified record of the original analysis environment. Exact package imports and versions must be checked against the analysis source code and the environment used to produce the reported results. Any additional package used for ALE, generalized additive models, hyperparameter search, file input, or notebooks must also be included if required by the uploaded code.

## Running the code

1. Download or clone the repository and extract any compressed data files as required by the scripts.
2. Install the dependencies in the intended Python environment.
3. Check the input filenames and paths in the uploaded scripts. Use paths relative to the repository root where possible.
4. Identify scripts whose inputs are fully available in the public release and run them in their documented order.
5. For scripts that require excluded geographic inputs, consult their input requirements and the corresponding methods in the manuscript.

Where sample-split labels or anonymous group identifiers are included, use the supplied assignments to maintain consistency with the original analysis. The original full-feature input table and analytical specification remain necessary for reproducing the complete manuscript results.

## Data provenance and reuse

The underlying platform data originate from the sources identified in the manuscript. This repository does not claim ownership of those original datasets. Any redistribution and reuse must comply with the applicable source-data permissions and terms.

This README does not assign a new license to third-party data. Code and data reuse conditions should be stated separately when the release is finalized.

## Citation

Please cite the accompanying manuscript when using these materials and acknowledge the original data sources described in the paper. Publication details and the article DOI can be added after publication.
