import itertools
import copy
import os
import pandas as pd


CSV = {
    "ml_feats": "hcc_ml.csv",
    "ml_os_labels": "hcc_labels_os.csv",
    "ml_grade_labels": "hcc_labels_grade.csv",
    "dl_feats": "hcc_dl.csv",
}


CORES = {
    "clinical": ['Sex','Cirrhosis','HBV','HCV','EtOH','NASH','PVT','M (EXTRAHEP METS)','ECOG PS baseline','Child-Pugh Score','AFP'],
    "treatment": ['Surgery','RFA/MWA','TACE','TARE','EBRT'],
    "morphological": ['grade','intratumoral lymphoid aggregates','histotype_nos','pattern_solid','VETC'],
}


FEATURE_SETS = {
    'clinical': CORES["clinical"],
    'treatment': CORES["treatment"],
    'morphological': CORES["morphological"],
    'clinical_treatment': CORES["clinical"] + CORES["treatment"],
    'clinical_morphological': CORES["clinical"] + CORES["morphological"],
    'treatment_morphological': CORES["treatment"] + CORES["morphological"],
    'clinical_treatment_morphological': CORES["clinical"] + CORES["treatment"] + CORES["morphological"]
}


# Feature sets evaluated in the paper, used when --feature_sets is not given
DEFAULT_FEATURE_SETS = {
    ('ml', 'os'): list(FEATURE_SETS),
    ('ml', 'grade'): list(FEATURE_SETS),
    ('dl_multi', 'os'): ['clinical', 'clinical_treatment', 'clinical_treatment_morphological'],
    ('dl_multi', 'grade'): ['morphological'],
}


ML_MODELS = ['lr', 'svm', 'naive', 'random', 'extra']
ML_SELECTORS = ['none', 'kbest', 'rfe', 'rfa', 'pca']
DL_LRS = [1e-5, 1e-3]
DL_WDS = [1e-3, 1e-1]


def get_feature_columns(feat_set_name, task, csv_path):
    """Columns of csv_path that belong to the feature set, in the order they appear in the csv."""

    keep_list = FEATURE_SETS[feat_set_name]

    # tumour grade is the target of the grade task, so it cannot be used as input
    if task == 'grade':
        keep_list = [f for f in keep_list if f != 'grade']

    all_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    missing = [f for f in keep_list if f not in all_columns]
    if missing:
        raise ValueError(f"Columns {missing} of feature set '{feat_set_name}' not found in {csv_path}")
    return [c for c in all_columns if c in keep_list]


def get_ml_configs(base_args):

    features_csv = os.path.join(base_args.csv_dir, CSV['ml_feats'])

    configs = []
    for f_set, model, selector in itertools.product(base_args.feature_sets, ML_MODELS, ML_SELECTORS):
        c = copy.copy(base_args)
        c.model = model
        c.selector = selector
        c.feature_set_name = f_set
        c.feature_columns = get_feature_columns(f_set, base_args.task, features_csv)
        c.features_csv = features_csv
        c.labels_csv = os.path.join(base_args.csv_dir, CSV[f'ml_{base_args.task}_labels'])
        c.results_root = os.path.join(base_args.results_dir, f'ml_{base_args.task}')
        configs.append(c)
    return configs


def get_dl_unimodal_configs(base_args):

    configs = []
    for model, lr, wd in itertools.product(['abmil', 'transmil', 'multimodal'], DL_LRS, DL_WDS):
        c = copy.copy(base_args)
        c.model, c.lr, c.wd = model, lr, wd
        c.use_clinical = False
        c.results_root = os.path.join(base_args.results_dir, f'dl_{base_args.task}_unimodal')
        configs.append(c)
    return configs


def get_dl_multimodal_configs(base_args):

    clinical_csv = os.path.join(base_args.csv_dir, CSV['dl_feats'])
    hyperparameters = [
        DL_LRS,
        DL_WDS,
        [True, False],  # fixed_dim (True=Fixed, False=Variable)
        ['sum', 'concat'] # fuse method
    ]

    configs = []
    for f_set in base_args.feature_sets:
        feature_columns = get_feature_columns(f_set, base_args.task, clinical_csv)
        for lr, wd, fixed_dim, fuse in itertools.product(*hyperparameters):
            c = copy.copy(base_args)
            c.model, c.lr, c.wd, c.fixed_dim, c.fuse = 'multimodal', lr, wd, fixed_dim, fuse
            c.use_clinical = True
            c.clinical_csv = clinical_csv
            c.feature_set_name = f_set
            c.feature_columns = feature_columns
            dim_str = "fixed" if fixed_dim else "var"
            c.results_root = os.path.join(base_args.results_dir, f'dl_{base_args.task}_multimodal', f_set, f'{dim_str}_{fuse}')
            configs.append(c)
    return configs
