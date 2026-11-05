# SMIB Dynamics Reconstruction: Method Comparison

## 1. Project objective

Compare the ability of three approaches to reconstruct the underlying
dynamics/governing equations of a single-machine infinite-bus (SMIB)
synchronous-generator system from noisy measurements:

1.  **MBAM + SINDy**
2.  **XGBoost-based reconstruction**
3.  **Autoencoder-based reconstruction**

The comparison concerns dynamics reconstruction and equation recovery
under noisy data. It does **not** include fault detection or fault
classification.

All methods must use the same underlying system, clean reference
trajectories, noisy observations, and train/validation/test partitions
wherever their input requirements permit.

## 2. Governing system

The reference plant is the fourth-order synchronous-generator (SG4)
model. Its implementation is currently represented by:

-   `sg4_parameters.m` --- model parameters.
-   `sg4_rhs.m` --- four-state differential equations.
-   `sg4_outputs.m` --- calculated generator outputs.
-   `test_sg4_simulation.m` --- disturbance simulation and plotting.

The four dynamic states are:

-   Rotor angle, `delta`
-   Rotor speed, `omega`
-   Direct-axis transient voltage, `eq_prime`
-   Quadrature-axis transient voltage, `ed_prime`

The current simulation applies a voltage disturbance from **0.3 s to 0.5
s** and simulates the response over **0--1 s**. The plotted quantities
include active power, reactive power, direct-axis current, and
quadrature-axis current.

The exact mathematical definitions and parameter values should be taken
from `sg4_rhs.m` and `sg4_parameters.m`. This README does not replace
those source files.

## 3. Current project status

### Completed

-   [x] SG4 parameter function is available.
-   [x] SG4 right-hand-side function is available.
-   [x] SG4 output calculation function is available.
-   [x] A disturbance simulation has been run and produced finite
    trajectories and output plots.
-   [x] The input interpolation issue in `sg4_rhs.m` was identified. The
    input signals should be interpolated using `interp1` rather than
    evaluated as functions of time.
-   [x] A governing-equation consistency check was run by comparing
    numerical trajectory derivatives with derivatives returned by
    `sg4_rhs.m`.

### Latest equation-consistency check

Reported RMSE values:

  State          Derivative residual RMSE
  ------------ --------------------------
  `delta`                  7.64177515e-04
  `omega`                  1.09393937e-05
  `eq_prime`               2.41227286e-06
  `ed_prime`               3.67727367e-04

These values indicate reasonable agreement between the numerical
derivatives and the model RHS for the tested trajectory. They are a
consistency check, not a complete physical validation or proof that the
model is correct.

The numerical differentiation method, sampling interval, and treatment
of disturbance switching instants affect these residuals. The validation
script and its exclusion window around switching times should be
retained with the results.

### Not yet completed

-   [ ] Finalize and save a reproducible clean baseline dataset.
-   [ ] Generate Gaussian-noise datasets.
-   [ ] Generate non-Gaussian-noise datasets.
-   [ ] Define and save common train/validation/test splits.
-   [ ] Implement or repair each reconstruction method.
-   [ ] Extract comparable governing-equation representations from
    methods that do not natively return symbolic equations.
-   [ ] Run the same evaluation protocol for all methods.
-   [ ] Produce comparison tables and plots.

## 4. Data-generation plan

### 4.1 Clean reference data

Generate clean SG4 trajectories from the governing equations. Preserve:

-   Time vector
-   Four state trajectories
-   Input/disturbance signals
-   Calculated outputs
-   Model parameters
-   Initial conditions
-   Simulation settings

Save the clean data separately and never overwrite it with noisy data.

### 4.2 Noisy observations

Create noisy copies of the clean measurements. Keep the clean
trajectories as reference targets for evaluation.

**Gaussian noise**

Use additive zero-mean Gaussian noise:

`y_noisy = y_clean + sigma * randn(size(y_clean))`

The noise level should be specified relative to each signal's scale,
rather than applying one absolute standard deviation to signals with
different units or magnitudes.

**Non-Gaussian noise**

At minimum, include these two cases:

1.  **Uniform noise:** additive noise drawn from a bounded uniform
    distribution.
2.  **Impulsive noise:** occasional large-amplitude deviations/outliers,
    with the outlier probability and amplitude explicitly recorded.

The distributions and noise levels must be documented. Use multiple
noise realizations for each noise condition so results do not depend on
one random draw.

### 4.3 Reproducibility

-   Set and record random-number-generator seeds.
-   Keep clean and noisy datasets in separate files or clearly named
    fields.
-   Store the noise distribution, noise level, seed, and affected
    signals with each dataset.
-   Do not generate separate underlying plant trajectories for different
    methods unless a separate experiment explicitly requires it.

## 5. Fair comparison protocol

Use identical clean reference data, noise realizations, and data
partitions for all methods when possible.

Recommended partitioning:

-   **Training set:** used to fit model parameters.
-   **Validation set:** used for hyperparameter selection and model
    selection.
-   **Test set:** held out until final evaluation.

Avoid randomly splitting adjacent time samples if that would allow
nearly identical neighboring samples to leak across partitions. Prefer
trajectory-level or contiguous time-block splits when the dataset
contains multiple trajectories.

Any method-specific preprocessing must be fitted using training data
only and then applied unchanged to validation and test data.

## 6. Method-specific roles

### MBAM + SINDy

-   Use MBAM to investigate reductions/limiting parameter regimes of the
    physical model.
-   Use SINDy to identify a sparse governing-equation representation
    from the available data.
-   Record the candidate library, scaling, derivative-estimation method,
    sparsity threshold, and selected terms.

### XGBoost

XGBoost is a predictive model, not inherently a symbolic
equation-discovery method. To compare governing-equation reconstruction,
define an explicit equation-extraction stage, such as
fitting/interpreting a constrained symbolic representation of its
predictions or using an agreed surrogate equation-identification
procedure.

Report predictive reconstruction and extracted-equation quality
separately.

### Autoencoder

An autoencoder learns a compressed representation and reconstruction
mapping; it does not inherently return governing equations. Define an
explicit dynamics/equation-identification stage using its latent
representation or reconstructed trajectories.

Report reconstruction quality and extracted-equation quality separately.

The equation-extraction stages for XGBoost and the autoencoder must be
specified before results are compared. Do not treat a good prediction
score alone as proof of correct equation recovery.

## 7. Evaluation metrics

Evaluate both trajectory reconstruction and governing-equation recovery.

### Trajectory reconstruction

-   State-wise RMSE and normalized RMSE
-   Output-wise reconstruction error
-   Error on held-out test trajectories
-   Robustness as noise level increases

### Governing-equation recovery

Where the true governing equations are known, evaluate:

-   Correct term/support recovery
-   False-positive and missed terms
-   Coefficient error after consistent scaling
-   Equation residual on held-out data
-   Complexity/sparsity of the recovered equations

Also record runtime and relevant computational settings.

## 8. Suggested folder structure

``` text
project/
├── README.md
├── src/
│   ├── sg4_parameters.m
│   ├── sg4_rhs.m
│   ├── sg4_outputs.m
│   └── methods/
├── scripts/
│   ├── test_sg4_simulation.m
│   ├── validate_sg4_equations.m
│   ├── generate_clean_data.m
│   ├── generate_noisy_data.m
│   └── run_comparison.m
├── data/
│   ├── clean/
│   └── noisy/
│       ├── gaussian/
│       ├── uniform/
│       └── impulsive/
├── results/
│   ├── figures/
│   ├── metrics/
│   └── models/
└── logs/
```

This is the proposed organization. Existing files may currently be in a
flat directory; move or update paths only when the project is ready for
that change.

## 9. Immediate next steps

1.  Confirm the current clean baseline data file and its variable names.
2.  Create a reproducible clean-data generation script.
3.  Create a noise-generation script supporting Gaussian, uniform, and
    impulsive noise.
4.  Generate several noise realizations at documented noise levels.
5.  Establish common train/validation/test partitions.
6.  Repair and run each method incrementally.
7.  Freeze the evaluation protocol before comparing final results.

## 10. Important working rules

-   Preserve the original source files before making substantial edits.
-   Change one issue at a time and run the relevant test after each
    change.
-   Do not silently change the SG4 equations or parameters to improve a
    method's score.
-   Keep a record of code changes, random seeds, settings, and outputs.
-   Distinguish equation recovery from trajectory prediction throughout
    the report.
