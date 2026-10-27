# Analysis Code

Source code accompanying *Operational Coordination and Instant Food Delivery Reliability: Evidence from Two Chinese Platform-City Settings*.

This is a code-only release. The order-level datasets, processed analysis tables, and data-derived outputs are not included. Running the analyses requires access to the corresponding cleaned input data. The analyses describe observational and model-implied associations, not causal effects or individual courier responsibility.

## Files and execution order

| File | Purpose | How it is used |
| --- | --- | --- |
| `main_experiments.py` | Main modelling, explanation, transfer, temporal-sensitivity, and scenario workflow. | Run first. It creates the shared sample assignments and selected model settings required by the supplementary analyses. |
| `supplementary_analyses.py` | Benchmark models, grouped resampling stability, robustness checks, and outcome descriptions. | Run after the main training stage has successfully generated its output files. Individual tasks can be selected from the command line. |
| `lightgbm_utils.py` | LightGBM utility functions and plotting routines for SHAP, interactions, one- and two-dimensional ALE, and local explanations. | Imported by `main_experiments.py`; keep it in the same directory. It is not the main experiment entry point. |

The main script imports `lightgbm_utils`. The supplementary script reads the main script's persisted files rather than importing and rerunning the main experiment.

## Main workflow

`main_experiments.py` contains seven sequential sections:

1. Model development and grouped validation: constructs date-courier-wave groups, persists development/test assignments and five development folds, performs Optuna tuning, fits final LightGBM models, and evaluates the held-out test samples.
2. SHAP, interaction, and one-dimensional ALE analysis, including figure-provenance and interaction-stability audits.
3. Two-dimensional ALE for selected feature pairs.
4. Fixed-horizon learning curves using the persisted grouped folds.
5. Local and cross-context evaluation on full test samples and common-support subsets, including development-set constant-bias calibration.
6. Strict temporal holdout sensitivity with expanding-window tuning.
7. Model-implied scenario screening for preparation-time changes, wave-duration caps, and combined caps.

The grouped main analysis uses a test fraction of 0.20 at the group level, five development folds, and 50 Optuna trials per setting. The temporal sensitivity section uses 30 trials. Settings are defined near the relevant sections of the script. A group-level test fraction does not force exactly 20% of order rows into the test sample.

The helper module also contains a standalone `train_lightgbm_optuna` function. The main workflow implements its own grouped training and model-selection protocol; running the helper function directly does not reproduce that protocol.

## Supplementary tasks

| `--task` | Analyses |
| --- | --- |
| `benchmark` | LightGBM, ordinary least squares, and a penalized B-spline additive benchmark. The additive benchmark uses scikit-learn's `SplineTransformer` and `Ridge`, not a separate GAM package. |
| `resampling` | Group-bootstrap model refits, held-out performance, SHAP rankings, ALE profiles, and selected SHAP interactions. Default: 50 refits per setting. |
| `robustness` | Removal of the wave-duration predictor and upper-tail trimming of wave duration/order load using development-sample 99th-percentile thresholds. An exact top-200 sensitivity is available only when a variable is explicitly selected. |
| `outcome` | Relative-delay distributions and descriptive statistics for relative delay, actual fulfillment duration, and promised fulfillment duration. |
| `all` | All four task families above; this is the default. The optional exact top-200 sensitivity still requires an explicit variable. |

## Input data contract

The scripts expect cleaned CSV analysis tables. They do not reconstruct the full analysis tables from raw platform event logs or perform raw coordinate restoration.

The response column is `standardized_delay`, representing the precomputed relative delivery-delay outcome used in the study. Its name does not imply that users should apply a new z-score transformation.

The shared predictor list is:

```text
order_hourly
is_weekend
peak
ride_distance_km
merchant_preparation_time
wave_duration_time
loadorder_number
courier_work_hours_now
platform_assign_time
demand_density
supply_density
road_length_km
avg_node_degree
prev_standardized_delay
```

Setting-specific input fields are:

| Requirement | Beijing (`BJ`) | Baoding (`BD`) |
| --- | --- | --- |
| Stable order identifier | `source_row_id` | `waybill_id` |
| Date-courier-wave grouping fields | `Date`, `courier_id`, `wave_index` | `dt`, `courier_id`, `wave_id` |
| Source column for the prior-order outcome predictor | `prev_standardized_delay` | `prior_order_delay` |

The Baoding input column `prior_order_delay` is renamed internally to `prev_standardized_delay`. Its values must follow the definition used in the paper; the scripts do not recompute it from raw delivery records.

The supplementary script additionally requires `finish_time` and `estimate_finish_time`. In these cleaned tables, both are numeric durations in minutes, not raw timestamp strings.

Order identifiers must be unique within each setting. Grouping keys must be complete. Wave identifiers alone are insufficient: the code constructs the analysis group from setting, date, courier, and wave. Input definitions, units, missing values, and source-row order should remain consistent with the original cleaned analysis tables and persisted assignments.

## Dependencies

Install the accompanying root-level `requirements.txt` from the repository root:

```bash
python -m pip install -r requirements.txt
```

The source imports NumPy, pandas, scikit-learn, LightGBM, Optuna, SHAP, Matplotlib, Seaborn, PyALE, and alibi. SciPy supports the numerical and clustering routines used by these packages. `alibi` is imported by the utility module and must be available even if a particular analysis does not call its ALE explainer.

The dependency list has been checked against the supplied source imports, but it is not a version-locked record of the original execution environment. In particular, the main script imports the internal PyALE function `PyALE._src.lib.quantile_ied`; PyALE compatibility should be checked in the execution environment. The benchmark pipelines also require a scikit-learn version supporting `sparse_output` in `OneHotEncoder` and `SplineTransformer`.

The main training stage records software versions in `results/software_environment.csv` under the configured output directory. Use the original run's record when preparing an exact environment for replication.

## Configure paths

Both executable scripts accept input and output locations through environment variables:

| Variable | Meaning |
| --- | --- |
| `DELIVERY_BJ_DATA` | Path to the cleaned Beijing CSV. |
| `DELIVERY_BD_DATA` | Path to the cleaned Baoding CSV. |
| `DELIVERY_OUTPUT_DIR` | Main output root. The supplementary script uses this same location to read shared assignments and selected model settings. |
| `DELIVERY_ADDITIONAL_OUTPUT_DIR` | Supplementary output root. It can also be set with `--output-root`. |
| `DELIVERY_FIGURE_FONT` | Font setting used by the supplementary figures and main scenario figures. Other plotting routines may retain their own font settings. |

Example in Windows PowerShell, run from the repository root after replacing the two input paths:

```powershell
$env:DELIVERY_BJ_DATA = "C:\delivery_data\beijing_cleaned.csv"
$env:DELIVERY_BD_DATA = "C:\delivery_data\baoding_cleaned.csv"
$env:DELIVERY_OUTPUT_DIR = Join-Path $PWD "outputs\grouped_5fold_lightgbm"
$env:DELIVERY_ADDITIONAL_OUTPUT_DIR = Join-Path $PWD "outputs\additional_analyses"
```

If the input variables are unset, the scripts look for `北京订单数据20261019.csv` and `保定订单数据20261019.csv` beside the scripts. These files are not included in the public repository. If the output variables are unset, outputs are written beneath the scripts' own `outputs` directory.

## Run the analyses

From the repository root, with the environment configured above:

```bash
python code/main_experiments.py
python code/supplementary_analyses.py --task all
```

The main script executes its sections at the top level and does not offer a task-selection command-line interface. Importing it as a library also triggers execution. Keep its dependent sections in order if running it interactively.

To run individual supplementary tasks:

```bash
python code/supplementary_analyses.py --task benchmark
python code/supplementary_analyses.py --task resampling
python code/supplementary_analyses.py --task robustness
python code/supplementary_analyses.py --task outcome
```

Additional examples:

```bash
python code/supplementary_analyses.py --help
python code/supplementary_analyses.py --task benchmark --cities BJ
python code/supplementary_analyses.py --task resampling --n-resamples 50
python code/supplementary_analyses.py --task robustness --tail200-variable loadorder_number
python code/supplementary_analyses.py --task robustness --tail200-variable wave_duration_time
python code/supplementary_analyses.py --task benchmark --smoke-test
```

The smoke-test option reduces the sample and computational settings, but still requires the cleaned data and main training outputs. Smoke-test results are for checking the workflow and must not be used as manuscript results.

## Persisted files and outputs

Before running supplementary tasks, the main training stage must have generated:

```text
<DELIVERY_OUTPUT_DIR>/
├── shared_sample_split/
│   ├── analysis_sample_split.csv
│   ├── analysis_wave_split.csv
│   └── split_metadata.json
└── models/
    ├── model_metadata.json
    ├── BJ/best_params.json
    └── BD/best_params.json
```

Retain the complete main output directory, including its model files. The scripts validate data/group consistency and reuse persisted assignments. Changing the inputs, their paths, or analysis settings can invalidate saved signatures and trigger rebuilding or refitting. Keep shared partitions and model settings from the same run together.

Main outputs include model files and metadata, held-out metrics, grouped learning curves, SHAP/ALE figures and audits, common-support transfer results, temporal sensitivity results, and model-implied scenario results. Supplementary outputs are organized into `01_benchmark_models`, `02_grouped_resampling_stability`, `03_robustness`, and `04_outcome_description` under the supplementary output root.

Existing complete resampling replicates are skipped. Use a separate output directory when changing resampling settings to avoid combining outputs from different configurations.

## Reproducibility and interpretation

Access to the same cleaned data, complete predictor set, sample assignments, model settings, and a compatible software environment is necessary to reproduce numerical results. The public code alone does not provide those data. Omitting geographic predictors changes the model specification and its explanations.

Preparation-time and wave-duration caps are scenario settings, not empirically established causal or abnormality thresholds. ALE zero crossings are centered statistical references. Scenario differences are conditional predictions from fitted models and should not be interpreted as demonstrated intervention effects.
