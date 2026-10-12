import argparse
import os
from experiments import FEATURE_SETS, DEFAULT_FEATURE_SETS, get_ml_configs, get_dl_unimodal_configs, get_dl_multimodal_configs
from machine_learning import run_ml_pipeline
from deep_learning import run_dl_pipeline


def get_args():
    parser = argparse.ArgumentParser(description='Prediction of overall survival and tumour grade in HCC from clinical data and whole-slide images')
    parser.add_argument('--mode', type=str, choices=['ml', 'dl_uni', 'dl_multi'], required=True,
                        help='ml: classical ML on tabular features; dl_uni: MIL on WSI features; dl_multi: WSI + tabular features')
    parser.add_argument('--task', type=str, choices=['os', 'grade'], required=True)
    parser.add_argument('--feature_sets', type=str, nargs='+', choices=list(FEATURE_SETS),
                        help='tabular feature sets to evaluate (ml and dl_multi). Default: the ones used in the paper')
    parser.add_argument('--csv_dir', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csvs'),
                        help='directory with the tabular features and the labels')
    parser.add_argument('--features_dir', type=str,
                        help='directory with one .pt file of patch features per slide (dl_uni and dl_multi)')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=48)
    parser.add_argument('--epochs', type=int, default=50, help='training epochs (dl_uni and dl_multi)')
    parser.add_argument('--save_shap', action='store_true', help='save the SHAP values of every LOOCV fold (ml only, slow)')
    parser.add_argument('--skip_done', action='store_true', help='skip the configurations whose results are already complete (to resume an interrupted run)')
    args = parser.parse_args()

    if args.mode != 'ml' and args.features_dir is None:
        parser.error('--features_dir is required for dl_uni and dl_multi')
    if args.mode != 'dl_uni' and args.feature_sets is None:
        args.feature_sets = DEFAULT_FEATURE_SETS[(args.mode, args.task)]
    return args


def main():
    args = get_args()

    if args.mode == 'ml':
        for c in get_ml_configs(args): run_ml_pipeline(c)

    elif args.mode == 'dl_uni':
        for c in get_dl_unimodal_configs(args): run_dl_pipeline(c)

    elif args.mode == 'dl_multi':
        for c in get_dl_multimodal_configs(args): run_dl_pipeline(c)

if __name__ == "__main__":
    main()
