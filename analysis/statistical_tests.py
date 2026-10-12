"""
Statistical comparison of the LOOCV macro F1 scores (LaTeX tables of the paper).

- ML vs ML feature sets: Wilcoxon signed-rank (paired by selector x classifier), Holm-Bonferroni.
- DL unimodal vs DL multimodal, and DL vs ML: Mann-Whitney U, Holm-Bonferroni.
  DL unimodal has 12 configurations (3 architectures x 4 hyperparameter pairs) and DL multimodal
  48 (OS) / 16 (grade) (4 fusion strategies x 4 hyperparameter pairs x feature sets), so there is
  no 1-to-1 pairing between configurations and an unpaired test is used.
- OS vs grade for each model family: Wilcoxon signed-rank, Holm-Bonferroni. Only families with the
  same configuration grid on both targets are paired (all ML families and DL unimodal); DL
  multimodal uses different feature sets for OS and grade and is skipped.

Usage: python analysis/statistical_tests.py --results_dir results --out_dir tables
"""

import argparse
import glob
import itertools
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu
from statsmodels.stats.multitest import multipletests


ML_FEATURE_SETS = ["clinical", "clinical_morphological", "clinical_treatment", "clinical_treatment_morphological",
                   "morphological", "treatment", "treatment_morphological"]
ML_SELECTORS = ["none", "kbest", "pca", "rfa", "rfe"]
ML_MODELS = ["lr", "svm", "naive", "random", "extra"]
DL_MULTIMODAL_FEATURE_SETS = {"os": ["clinical", "clinical_treatment", "clinical_treatment_morphological"],
                              "grade": ["morphological"]}


def extract_metric_from_txt(file_path, pattern):
    with open(file_path, "r") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                return float(match.group(1))
    return None


def ml_label(feature_set):
    """clinical_treatment -> 'ML (C+T)'"""
    parts = [abbr for word, abbr in [("clinical", "C"), ("treatment", "T"), ("morphological", "M")] if word in feature_set]
    return f"ML ({'+'.join(parts)})"


def parse_ml_results(base_dir):
    # fixed iteration order: the Wilcoxon test pairs the configurations of two feature sets by position
    records = []
    for feature_set, selector, model in itertools.product(ML_FEATURE_SETS, ML_SELECTORS, ML_MODELS):
        file_path = os.path.join(base_dir, feature_set, selector, f"{model}.txt")
        if not os.path.exists(file_path):
            print(f"[WARNING] Missing: {file_path}")
            continue
        value = extract_metric_from_txt(file_path, r"Final LOOCV macro F1:\s*([0-9.]+)")
        if value is not None:
            records.append({"Model": ml_label(feature_set), "F1-Macro Score": value})
    return pd.DataFrame(records)


def parse_dl_results(base_dir, label):
    records = []
    for file_path in sorted(glob.glob(os.path.join(base_dir, "*.txt"))):
        value = extract_metric_from_txt(file_path, r"Macro F1:\s*([0-9.]+)")
        if value is not None:
            records.append({"Model": label, "F1-Macro Score": value})
    if not records:
        print(f"[WARNING] No results in {base_dir}")
    return pd.DataFrame(records)


def load_scores(results_dir):
    dfs = []
    for task, target in [("os", "OS"), ("grade", "GRADE")]:
        multimodal_dir = os.path.join(results_dir, f"dl_{task}_multimodal")
        dfs += [
            parse_ml_results(os.path.join(results_dir, f"ml_{task}")).assign(Target=target),
            parse_dl_results(os.path.join(results_dir, f"dl_{task}_unimodal"), "DL Unimodal").assign(Target=target),
        ]
        dfs += [parse_dl_results(fusion_dir, "DL Multimodal").assign(Target=target)
                for feature_set in DL_MULTIMODAL_FEATURE_SETS[task]
                for fusion_dir in sorted(glob.glob(os.path.join(multimodal_dir, feature_set, "*")))]
    return pd.concat(dfs, ignore_index=True)


def stars(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'ns'


def fmt_p(p):
    return f"{p:.4f}" if p >= 0.0001 else "<0.0001"


def holm(df, by='Target'):
    for target in df[by].unique():
        mask = df[by] == target
        _, p_corr, _, _ = multipletests(df.loc[mask, 'p_raw'], alpha=0.05, method='holm')
        df.loc[mask, 'p_holm'] = p_corr
    df['Sig'] = df['p_holm'].apply(stars)
    return df


# ---------- ML vs ML (Wilcoxon signed-rank + Holm) ----------
def ml_pairwise(df, target):
    df_f = df[(df['Target'] == target) & (df['Model'].str.startswith('ML'))]
    models = sorted(df_f['Model'].unique())
    rows = []
    for m1, m2 in itertools.combinations(models, 2):
        s1 = df_f[df_f['Model'] == m1]['F1-Macro Score'].values
        s2 = df_f[df_f['Model'] == m2]['F1-Macro Score'].values
        if len(s1) == len(s2) and len(s1) > 0 and not np.all(s1 == s2):
            _, p = wilcoxon(s1, s2)
        else:
            p = np.nan
        rows.append({'Target': target, 'Model A': m1, 'Model B': m2, 'p_raw': p, 'Test': 'Wilcoxon signed-rank'})
    return pd.DataFrame(rows)


# ---------- DL Uni vs DL Multi, and DL vs ML (Mann-Whitney U + Holm) ----------
def mwu_p(df, target, mA, mB):
    sA = df[(df['Target'] == target) & (df['Model'] == mA)]['F1-Macro Score'].values
    sB = df[(df['Target'] == target) & (df['Model'] == mB)]['F1-Macro Score'].values
    _, p = mannwhitneyu(sA, sB, alternative='two-sided')
    return p


def dl_comparisons(df):
    ml_models = sorted(df[df['Model'].str.startswith('ML')]['Model'].unique())
    dl_models = ['DL Unimodal', 'DL Multimodal']
    rows = []
    for target in ['OS', 'GRADE']:
        rows.append({'Target': target, 'Model A': 'DL Unimodal', 'Model B': 'DL Multimodal',
                     'p_raw': mwu_p(df, target, 'DL Unimodal', 'DL Multimodal'), 'Test': 'Mann-Whitney U'})
        for dl in dl_models:
            for ml in ml_models:
                rows.append({'Target': target, 'Model A': dl, 'Model B': ml,
                             'p_raw': mwu_p(df, target, dl, ml), 'Test': 'Mann-Whitney U'})
    return pd.DataFrame(rows)


# ---------- OS vs GRADE for each model family (Wilcoxon signed-rank + Holm) ----------
def compare_os_vs_grade(df):
    rows = []
    for model in sorted(df['Model'].unique()):
        s_os = df[(df['Model'] == model) & (df['Target'] == 'OS')]['F1-Macro Score'].values
        s_grade = df[(df['Model'] == model) & (df['Target'] == 'GRADE')]['F1-Macro Score'].values
        if len(s_os) == len(s_grade) and len(s_os) > 0 and not np.all(s_os == s_grade):
            _, p = wilcoxon(s_os, s_grade)
        else:
            p = np.nan
        rows.append({'Model': model, 'Test': 'Wilcoxon signed-rank', 'n_OS': len(s_os), 'n_Grade': len(s_grade), 'p_raw': p})
    out = pd.DataFrame(rows)
    valid = out['p_raw'].notna()
    _, p_corr, _, _ = multipletests(out.loc[valid, 'p_raw'], alpha=0.05, method='holm')
    out.loc[valid, 'p_holm'] = p_corr
    out['Sig'] = out['p_holm'].apply(lambda p: stars(p) if pd.notna(p) else '--')
    skipped = out[~valid]['Model'].tolist()
    if skipped:
        print(f"OS vs grade comparison skipped (different number of configurations) for: {skipped}")
    return out


# ---------- Wide table (OS and grade side by side) ----------
def build_wide(df):
    df = df.copy()
    df['Comparison'] = df['Model A'] + ' vs ' + df['Model B']
    out = []
    for comp in df['Comparison'].unique():
        row = {'Comparison': comp, 'Test': df[df['Comparison'] == comp]['Test'].iloc[0]}
        for target, plabel, slabel in [('OS', 'p (OS)', 'Sig. (OS)'), ('GRADE', 'p (Grade)', 'Sig. (Grade)')]:
            sub = df[(df['Comparison'] == comp) & (df['Target'] == target)]
            if len(sub):
                row[plabel] = fmt_p(sub['p_holm'].iloc[0])
                row[slabel] = sub['Sig'].iloc[0]
            else:
                row[plabel] = '--'
                row[slabel] = '--'
        out.append(row)
    return pd.DataFrame(out)


# ---------- LaTeX export ----------
def to_latex_table(df, caption, label):
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append("Comparison & \\multicolumn{2}{c}{OS} & \\multicolumn{2}{c}{Tumor Grade} \\\\")
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
    lines.append("& $p$ (Holm) & Sig. & $p$ (Holm) & Sig. \\\\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        lines.append(f"{row['Comparison']} & {row['p (OS)']} & {row['Sig. (OS)']} & {row['p (Grade)']} & {row['Sig. (Grade)']} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def to_latex_single_table(df, caption, label):
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append("Model configuration & $p$ (Holm) & Significance \\\\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        p_str = fmt_p(row['p_holm']) if pd.notna(row['p_holm']) else '--'
        lines.append(f"{row['Model']} & {p_str} & {row['Sig']} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results_dir', default='results', help='--results_dir used with main.py')
    parser.add_argument('--out_dir', default='tables')
    args = parser.parse_args()

    df_final = load_scores(args.results_dir)
    print(df_final.groupby(['Target', 'Model']).size())

    ml_all = holm(pd.concat([ml_pairwise(df_final, 'OS'), ml_pairwise(df_final, 'GRADE')], ignore_index=True))
    mwu_all = holm(dl_comparisons(df_final))
    os_vs_grade = compare_os_vs_grade(df_final)

    latex_ml = to_latex_table(
        build_wide(ml_all),
        "Pairwise comparison of ML feature-set configurations (Wilcoxon signed-rank test, Holm--Bonferroni corrected within each target).",
        "tab:ml_pvalues"
    )
    latex_mwu = to_latex_table(
        build_wide(mwu_all),
        "Comparison between DL Unimodal/Multimodal and between DL and ML configurations (Mann--Whitney U test, Holm--Bonferroni corrected within each target).",
        "tab:dl_ml_pvalues"
    )
    latex_os_vs_grade = to_latex_single_table(
        os_vs_grade,
        "Performance comparison between overall survival and tumor grade prediction tasks. Statistical "
        "significance was assessed using the Wilcoxon signed-rank test with Holm--Bonferroni correction for "
        "multiple comparisons, for each model configuration (ML = Machine Learning, DL = Deep Learning; "
        "C = clinical, T = treatment, M = morphological).",
        "tab:os_vs_grade_pvalues"
    )

    os.makedirs(args.out_dir, exist_ok=True)
    for name, table in [("table_ml_pvalues.tex", latex_ml), ("table_dl_ml_pvalues.tex", latex_mwu),
                        ("table_os_vs_grade_pvalues.tex", latex_os_vs_grade)]:
        print(f"\n{table}")
        with open(os.path.join(args.out_dir, name), "w") as f:
            f.write(table)


if __name__ == "__main__":
    main()
