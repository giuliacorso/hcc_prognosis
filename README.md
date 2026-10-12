# HCC prognosis: prediction of overall survival and tumour grade

Code for the paper *TODO: title, authors, journal, DOI*.

The repository compares, on a single-centre cohort of hepatocellular carcinoma (HCC) biopsies:

- **ML**: classical machine learning on tabular clinical, treatment and morphological features;
- **DL unimodal**: multiple instance learning (MIL) on patch features of the whole-slide images (WSIs);
- **DL multimodal**: WSI patch features fused with the tabular features;

on two binary tasks: **overall survival** (OS > 18 months) and **tumour grade** (G1-2 vs G3-4).

## Repository structure

```
main.py                 entry point of the experiments
experiments.py          feature sets and hyperparameter grids of every experiment
machine_learning.py     nested LOOCV for the ML models
deep_learning.py        nested LOOCV for the MIL / multimodal models
dataset.py              loader of the WSI patch features
models/                 ABMIL, TransMIL and the multimodal model
analysis/
  statistical_tests.py  statistical comparison of the results (LaTeX tables)
```

## Installation

```bash
pip install -r requirements.txt
```

The results of the paper were produced with Python 3.10 and the versions pinned in `requirements.txt`.

## Data

The patient data are not included in this repository. The code expects the following inputs.

### Tabular data (`--csv_dir`, default `csvs/`)

| File | Content |
|---|---|
| `hcc_ml.csv` | one row per patient: `patient_id` and the tabular features (input of the ML experiments) |
| `hcc_labels_os.csv` | `patient_id`, `label`: OS label (`1` = OS > 18 months) of the patients of the experiments |
| `hcc_labels_grade.csv` | `patient_id`, `label`: tumour grade label (`1` = G3-4) of the same patients |
| `hcc_dl.csv` | one row per slide: `slide_id` and the same tabular features (input of the DL multimodal experiments) |

Patients alive with less than 18 months of follow-up are excluded from both tasks, because their OS class is unknown. All the features are binary.

| Feature set | Columns |
|---|---|
| clinical (C) | Sex, Cirrhosis, HBV, HCV, EtOH, NASH, PVT, M (EXTRAHEP METS), ECOG PS baseline, Child-Pugh Score, AFP |
| treatment (T) | Surgery, RFA/MWA, TACE, TARE, EBRT |
| morphological (M) | grade, intratumoral lymphoid aggregates, histotype_nos, pattern_solid, VETC |

Child-Pugh score is 0 for A5 and 1 for A6, AFP is 1 above 400 ng/mL, and grade is 0 for G1-2 and 1 for G3-4. All 7 combinations (C, T, M, C+T, C+M, T+M, C+T+M) are defined in `experiments.py`. For the grade task, `grade` is removed from the inputs because it is the target. Other columns of the tables are ignored.

### WSI features (`--features_dir`)

One `.pt` file per slide, containing a dict with:

- `features`: tensor `n_patches x 1536`;
- `label`: tensor with the class of the slide for the task;
- `slide_id`: the slide id, matching `slide_id` in `hcc_dl.csv`.

The features were extracted with UNI2-h from 256x256 px patches at 20x. Since the label is stored in the files, OS and grade need separate feature directories.

## Running the experiments

Every run evaluates the full hyperparameter grid with nested leave-one-out cross-validation. Results are written to `--results_dir` (default `results/`).

```bash
# ML: 7 feature sets x 5 feature selectors x 5 classifiers
python main.py --mode ml --task os --save_shap
python main.py --mode ml --task grade

# DL unimodal: ABMIL, TransMIL and the multimodal model on WSI only (x 2 learning rates x 2 weight decays)
python main.py --mode dl_uni --task os --features_dir /path/to/features_os
python main.py --mode dl_uni --task grade --features_dir /path/to/features_grade

# DL multimodal: 4 fusion strategies x 2 learning rates x 2 weight decays, for each feature set
python main.py --mode dl_multi --task os --features_dir /path/to/features_os
python main.py --mode dl_multi --task grade --features_dir /path/to/features_grade
```

When `--feature_sets` is not given, the feature sets used in the paper are evaluated:
- ML: all 7;
- DL multimodal OS: C, C+T and C+T+M;
- DL multimodal grade: M.

Use `python main.py --help` for the other options. `--save_shap` saves the SHAP values of every LOOCV fold and makes the runs much slower. With `--skip_done`, the configurations whose results are already complete are skipped, so an interrupted run can be resumed by launching it again.

### Methods in brief

- **ML**: outer LOOCV. In each fold, a grid search with stratified 3-fold CV (macro F1) runs over a pipeline of:
  - an optional feature selector: none, SelectKBest with mutual information, RFE, recursive feature addition, or PCA;
  - a classifier: logistic regression, SVM, Bernoulli naive Bayes, random forest, or extra trees.
- **DL**: outer LOOCV. In each fold, a model is trained on each of the 3 stratified inner folds (Adam, cosine annealing, 50 epochs). For each inner fold the epoch with the best validation macro F1 is kept, and the best of the 3 models predicts the held-out patient.
- **Multimodal model**:
  - gated attention pooling of the WSI patches with learnable latent queries;
  - an MLP encoder for the tabular features;
  - fusion by sum or concatenation;
  - embedding dimension either fixed (64, `fixed`) or equal to the number of tabular features (`var`).
- **Metric:** macro F1 over the LOOCV predictions. Seed: 48.

## Results

```
results/
  ml_<task>/<feature set>/<selector>/<model>.txt           F1, confusion matrix, misclassified patients
  ml_<task>/<feature set>/<selector>/<model>.csv           features selected in each fold
  ml_<task>_shap/<feature set>/<model>/<selector>/fold_N/  SHAP values (--save_shap)
  dl_<task>_unimodal/<model>_<lr>_<wd>.txt
  dl_<task>_multimodal/<feature set>/<fixed|var>_<sum|concat>/multimodal_<lr>_<wd>.txt
```

The statistical comparison of the paper (Wilcoxon signed-rank and Mann-Whitney U tests with Holm-Bonferroni correction) is computed from these results:

```bash
python analysis/statistical_tests.py --results_dir results --out_dir tables
```

## Reproducibility

All random seeds are fixed. Results of the DL models may vary slightly on different GPUs or library versions because of non-deterministic CUDA kernels.

## Acknowledgements

- ABMIL: Ilse, Tomczak and Welling, *Attention-based Deep Multiple Instance Learning*, ICML 2018.
- TransMIL: Shao et al., *TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification*, NeurIPS 2021.

## License

*TODO*
