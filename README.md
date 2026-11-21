# Punching shear of RC flat slabs: does machine learning beat Eurocode 2 on unseen test programmes?

[![CI](https://github.com/hildalakzaei7-pixel/punching-shear-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/hildalakzaei7-pixel/punching-shear-ml/actions)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/code-MIT-green)

Punching at slab–column connections is brittle, and it governs the design of many flat slabs.
Test databases are a natural target for machine learning, but their tests come in series: one
laboratory programme changes one or two parameters across specimens that are otherwise identical.
A random train/test split therefore puts near-twins on both sides and measures how well a model
interpolates *within* programmes.

This project re-asks the question under the condition that matters in practice:

> **How well does each model predict a test programme it has never seen?**

| | |
|---|---|
| **Data** | 228 interior slab–column tests without shear reinforcement, from 41 test programmes (TU Dortmund database) |
| **Baseline** | EN 1992-1-1:2004, 6.4.4, with mean material values |
| **Validation** | 5-fold cross-validation grouped by test programme, 5 repeats, nested tuning |
| **Main result** | Random splits make tree models look as good as the code; on unseen programmes they are clearly worse |
| **Best on unseen programmes** | A 6-parameter mechanics-based power law and an EC2-anchored XGBoost correction, both with CoV ≈ 0.19–0.20 |

---

## 1. Data

The source is the open database by Spyridis, Mellios, Uz and Walter (2023): 543 full-scale tests from
54 publications. It is protected by original copyright, so **it is not redistributed here**; see
[`data/README.md`](data/README.md) for the download and the checksum the code verifies.

Cleaning (full rationale in [`data/DATA_DICTIONARY.md`](data/DATA_DICTIONARY.md)):

* kept interior connections without shear reinforcement and without fibres;
* removed tests reported as flexural failures;
* kept tests whose failure mode is not reported (they include classic punching series), flagged, and
  re-ran the key comparison without them (section 6);
* left out the steel yield strength as an input (21 % missing in the core set; the EC2 expression does not use it).

| Input | Range in the core set |
|---|---|
| effective depth *d* | 39–590 mm |
| column size *b* (square side or circle diameter) | 54–520 mm |
| concrete strength *f*<sub>c</sub> (mean cylinder) | 13–119 MPa |
| flexural reinforcement ratio ρ | 0.2–3.7 % |
| load radius *r* (span proxy) | 325–2580 mm |

![Distributions](results/figures/01_distributions.png)
![Tests per programme](results/figures/02_tests_per_programme.png)

## 2. Models

| Model | Idea | Fitted parameters |
|---|---|---|
| **EC2 (mean values)** | *v* = 0.18 *k* (100 ρ *f*<sub>c</sub>)<sup>1/3</sup> ≥ *v*<sub>min</sub>, *V* = *v u*<sub>1</sub> *d*, *k* ≤ 2, ρ ≤ 2 % | none |
| EC2 (C refitted) | same form, constant 0.18 refitted | 1 |
| **Power law** | ln *V* linear in ln *d*, ln *f*<sub>c</sub>, ln ρ, ln *u*<sub>1</sub>, ln(*r*/*d*) | 6 |
| Random forest, XGBoost | trees on raw inputs + dimensionless groups, target ln *V* | many |
| XGBoost (stress target) | trees on the nominal stress *V* / (*u*<sub>1</sub> *d*) | many |
| **EC2 + XGBoost correction** | *V* = *V*<sub>EC2</sub> · exp(*f*(x)); trees learn only the code's error | many |

Everything except the EC2 baseline is trained in log space, because test scatter grows with capacity.
Engineering accuracy is reported as the ratio *V*<sub>test</sub>/*V*<sub>pred</sub>: its mean (bias),
its coefficient of variation (scatter), and the share of tests with a ratio below 1 (prediction above
the measured failure load).

## 3. Main result: random splits reverse the ranking

![Random vs grouped CV](results/figures/03_random_vs_grouped.png)

| Model | R², random | R², grouped | CoV of ratio, grouped | Mean ratio, grouped |
|---|---:|---:|---:|---:|
| EC2 (mean values) | 0.916 | 0.916 | 0.196 | 1.17 |
| EC2 (C refitted) | 0.932 | 0.930 | 0.202 | 1.02 |
| **Power law** | 0.959 | **0.949** | **0.196** | 1.01 |
| **EC2 + XGBoost correction** | 0.957 | **0.934** | **0.194** | 1.03 |
| XGBoost (stress target) | 0.931 | 0.870 | 0.224 | 1.05 |
| XGBoost | 0.876 | 0.609 | 0.339 | 1.11 |
| Random forest | 0.817 | 0.548 | 0.410 | 1.12 |

*Mean of 5 repeats × 5 folds. The EC2 baseline has no fitted parameters, so both protocols give the
same numbers. With hyper-parameters tuned inside each outer fold only (nested), the EC2 + XGBoost
correction gives CoV 0.190 ± 0.017.*

What this shows:

1. **Random cross-validation hides a failure.** Plain XGBoost reaches R² = 0.88 on random folds and
   0.61 on unseen programmes.
2. **The cause is extrapolation in size.** Trees cannot predict outside the force range they were
   trained on. When the thickest slabs (*d* up to 590 mm) are held out, XGBoost under-predicts them by
   a factor of up to 3:

   ![Ratio vs depth](results/figures/04_ratio_vs_depth.png)

3. **Normalising the target helps but does not close the gap.** Training on nominal stress raises R²
   to 0.87, still below the code.
4. **On new programmes, data-driven models do not beat a calibrated code expression.** The best
   models reach CoV ≈ 0.19–0.20, which is the scatter of the EC2 expression itself. Within this
   database, the ML correction mainly removes EC2's conservative bias (mean ratio 1.17 → 1.03); it
   does not reduce the scatter.

## 4. What the data say about the EC2 expression

The power law is refitted on 500 bootstrap resamples of *whole programmes*:

![Exponents](results/figures/05_powerlaw_exponents.png)

* The fitted exponents of *f*<sub>c</sub> (0.37, 90 % CI 0.27–0.42) and ρ (0.37, CI 0.31–0.43) sit
  close to the 1/3 used in EC2.
* Slenderness *r*/*d*, absent from the EC2 expression, has a negative exponent whose interval
  excludes zero (−0.26, CI −0.42 to −0.15) once the other variables are held fixed.
* The *d* and *u*<sub>1</sub> exponents are poorly identified (wide intervals): depth and perimeter
  are strongly correlated in the tests.

The residuals of EC2 line up with its own caps:

![EC2 residuals](results/figures/07_ec2_residuals.png)

* **Thin slabs.** For *d* < 200 mm, where *k* is capped at 2.0, the mean ratio is 1.21; for
  *d* ≥ 200 mm it is 1.07.
* **Heavy reinforcement.** Above the ρ ≤ 2 % cap, the mean ratio is 1.21 (22 tests).
* **Stocky slabs.** For *r*/*d* < 4 the mean ratio is 1.30, but these are only 14 tests from 4
  programmes. This is an observation, not a finding.

The SHAP analysis of the EC2-correction model points the same way: slenderness, load radius, ρ and
*d* are the inputs on which the code's error depends most.

![SHAP](results/figures/06_shap_ec2_correction.png)

## 5. Uncertainty on unseen programmes

Split conformal prediction (Papadopoulos et al., 2002; Lei et al., 2018), applied on the log scale
with training, calibration and test sets disjoint **by programme**, over 30 random group splits:

| Model | Mean coverage (target 90 %) | Interval |
|---|---:|---|
| EC2 (C refitted) | 90.6 % ± 10.5 | *V̂* ÷ 1.47 to *V̂* × 1.47 |
| Power law | 93.9 % ± 4.5 | *V̂* ÷ 1.54 to *V̂* × 1.54 |
| EC2 + XGBoost correction | 92.8 % ± 5.8 | *V̂* ÷ 1.46 to *V̂* × 1.46 |

A factor of about 1.5 either way is the honest width for a 90 % interval on a new test programme.
Coverage varies a lot between splits because whole programmes move in and out of the calibration set.

![Parity](results/figures/08_parity_powerlaw.png)

## 6. Sensitivity

Restricted to the 141 tests (28 programmes) whose failure mode is reported as punching, the
conclusions hold. Grouped CoV is 0.205 for EC2, 0.191 for the power law and 0.184 for the EC2 + XGBoost
correction; EC2's mean ratio rises to 1.21.

## 7. Limitations

* One database, one test type: concentric loading, interior columns, no shear reinforcement.
  Edge and corner columns, moment transfer, shear reinforcement and fibres are not covered.
* EC2 is evaluated with mean values and γ<sub>c</sub> = 1. This is a comparison of models with tests,
  **not a design check**, and the ratios say nothing about the safety of the code with its partial factors.
* Circular columns: the database gives the same equivalent diameter formula (*b*′ = 1.128 *b*) for
  square and circular columns. The code takes *b* as the diameter of circular columns; this
  interpretation could not be checked against the original publications.
* 41 programmes is a small number of independent groups; the spread across repeats is reported for that reason.
* No external validation yet. The fib / ACI 445 punching database is the natural next test set.

## Reproduce

```bash
git clone https://github.com/hildalakzaei7-pixel/punching-shear-ml.git
cd punching-shear-ml
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
# download the workbook into data/raw/ (see data/README.md), then:
pytest -q                          # 9 tests; the last one checks the real file if present
python -m punching_ml.pipeline     # about 3 minutes; writes results/
```

```
src/punching_ml/
├── data.py       parsing, programme IDs, cleaning rules, checksum
├── ec2.py        EN 1992-1-1 6.4.4 with mean values
├── models.py     power law, EC2 correction, tree models
├── evaluate.py   random vs grouped CV, nested tuning, ratio metrics, grouped conformal
├── plots.py      figure style
└── pipeline.py   full experiment → results/
```

## References

1. Spyridis, P., Mellios, N., Uz, O., & Walter, L. (2023). *Database of experimental results from punching
   shear tests with various types of reinforcement* [Dataset]. TU Dortmund. https://doi.org/10.17877/DE290R-23085
2. Mellios, N., Uz, O., & Spyridis, P. (2023). Data-based modeling of the punching shear capacity of concrete
   structures. *Engineering Structures*, 275, 115195.
3. CEN (2004). *EN 1992-1-1: Eurocode 2: Design of concrete structures. Part 1-1: General rules and rules
   for buildings*, clause 6.4.4.
4. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proc. 22nd ACM SIGKDD*,
   785–794. https://doi.org/10.1145/2939672.2939785
5. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions.
   *Advances in Neural Information Processing Systems 30*, 4765–4774. https://arxiv.org/abs/1705.07874
6. Papadopoulos, H., Proedrou, K., Vovk, V., & Gammerman, A. (2002). Inductive confidence machines for
   regression. *Machine Learning: ECML 2002*, LNCS 2430, 345–356. https://doi.org/10.1007/3-540-36755-1_29
7. Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). Distribution-free predictive
   inference for regression. *JASA*, 113(523), 1094–1111. https://doi.org/10.1080/01621459.2017.1307116

## Author

**Hilda Lakzaei**, Civil Engineering (Structural), Sharif University of Technology.
Related: [concrete-strength-ml](https://github.com/hildalakzaei7-pixel/concrete-strength-ml), which uses the same grouped-validation approach.
[LinkedIn](https://www.linkedin.com/in/hilda-lakzaei-32a4a8341) · [Website](https://hilda-lakzaei.netlify.app/)
