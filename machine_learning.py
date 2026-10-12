import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from feature_engine.selection import RecursiveFeatureAddition
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import shap

from seed import init_seed


def load_data(args):
    """Feature matrix and labels of the patients listed in the labels csv, in that order."""

    labels = pd.read_csv(args.labels_csv)
    features = pd.read_csv(args.features_csv).set_index('patient_id')[args.feature_columns]
    missing = set(labels['patient_id']) - set(features.index)
    if missing:
        raise ValueError(f"Patients without features in {args.features_csv}: {sorted(missing)}")
    names = labels['patient_id'].tolist()
    return features.loc[names].values, labels['label'].values, names, list(features.columns)


def run_ml_pipeline(args):

    loo = LeaveOneOut()
    init_seed(args)
    results_list = []
    results_path = f"{args.results_root}/{args.feature_set_name}/{args.selector}/{args.model}.txt"
    if args.skip_done and _is_done(results_path, "Final LOOCV macro F1"):
        print(f"Skipping {results_path}: already done")
        return
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    with open(results_path, "w") as f:
        f.write(f"\nRunning nested cross-validation with model {args.model} and selector {args.selector}\n")

    X, y, names, feature_names = load_data(args)
    y_true_all, y_pred_all, total_wrong = [], [], []

    for fold, (train_idx, test_idx) in enumerate(loo.split(X)):
        np.random.seed(args.seed + fold)

        X_trainval, y_trainval = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        model_dict = {
            'lr': LogisticRegression(max_iter=5000, random_state=args.seed),
            'svm': SVC(max_iter=5000, random_state=args.seed, probability=True),
            'naive': BernoulliNB(),
            'random': RandomForestClassifier(random_state=args.seed),
            'extra': ExtraTreesClassifier(random_state=args.seed)
        }
        base_clf = model_dict.get(args.model)
        if base_clf is None:
            raise ValueError(f"Unsupported model: {args.model}")

        pipeline, param_grid = _build_pipeline_and_grid(args, base_clf, X_trainval.shape[1])
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
        grid = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring='f1_macro', n_jobs=1)
        grid.fit(X_trainval, y_trainval)
        best_model = grid.best_estimator_

        preds = best_model.predict(X_test)
        f1 = f1_score(y_test, preds, average='macro')
        y_true_all.extend(y_test)
        y_pred_all.extend(preds)

        if preds[0] != y_test[0]:
            total_wrong.append(names[test_idx[0]])

        if args.selector != 'none':
            if args.selector == 'pca':
                support = list(range(X_trainval.shape[1]))
            else:
                selector_step = next(s for s in best_model.named_steps if hasattr(best_model.named_steps[s], 'get_support'))
                support = best_model.named_steps[selector_step].get_support(indices=True)
            fold_result = {
                "fold": fold + 1,
                "selector_type": args.selector,
                "classifier": args.model,
                "f1_score": f1,
                "n_selected_features": len(support),
                "selected_feature_indices": ",".join(map(str, support)),
                "selected_feature_names": ",".join(feature_names[i] for i in support)
            }
            results_list.append(fold_result)
            print(fold_result)

        if args.save_shap:
            _save_shap_values(args, best_model, X_trainval, X_test, fold, feature_names)

    final_f1 = f1_score(y_true_all, y_pred_all, average='macro')
    cm = confusion_matrix(y_true_all, y_pred_all)
    with open(results_path, "a") as f:
        f.write(f"\nMisclassified: {total_wrong}\nFinal LOOCV macro F1: {final_f1:.4f}\nConfusion Matrix:\n{cm}\n")
    if args.selector != 'none':
        pd.DataFrame(results_list).to_csv(os.path.splitext(results_path)[0] + '.csv', index=False)

    print(f"LOOCV completed — Final F1: {final_f1:.4f}")


def _is_done(results_path, marker):
    if not os.path.exists(results_path):
        return False
    with open(results_path) as f:
        return marker in f.read()


def _build_pipeline_and_grid(args, base_clf, n_features):

    selector_params, model_params = {}, {}

    steps = []

    if args.selector == 'none':
        steps.append(('clf', base_clf))
    elif args.selector == 'kbest':
        steps.append(('selectkbest', SelectKBest(score_func=mutual_info_classif)))
        steps.append(('clf', base_clf))
        selector_params = {'selectkbest__k': list(range(1, n_features + 1))}
    elif args.selector == 'rfe':
        steps.append(('rfe', RFE(base_clf)))
        steps.append(('clf', base_clf))
        selector_params = {'rfe__n_features_to_select': list(range(1, n_features + 1))}
    elif args.selector == 'rfa':
        steps.append(('rfa', RecursiveFeatureAddition(base_clf, scoring='f1_macro')))
        steps.append(('clf', base_clf))
        selector_params = {'rfa__threshold': [0.0001, 0.001, 0.01]}
    elif args.selector == 'pca':
        steps.append(('pca', PCA(random_state=args.seed)))
        steps.append(('clf', base_clf))
        selector_params = {'pca__n_components': [2, 3, 4]}
    else:
        raise ValueError(f"Unsupported selector: {args.selector}")

    pipeline = Pipeline(steps)

    if args.model == 'lr':
        model_params = {"clf__penalty": ['l1', 'l2'],
                        "clf__solver": ['liblinear', 'saga'],
                        "clf__C": [0.01, 0.1, 1, 10]}
    elif args.model == 'svm':
        model_params = {"clf__C": [0.01, 0.1, 1, 10],
                        "clf__kernel": ["linear", "rbf"],
                        "clf__gamma": ["scale", "auto"]}
    elif args.model == 'naive':
        model_params = {"clf__alpha": [0.1, 0.5, 1.0, 2.0, 5.0],
                        "clf__fit_prior": [True, False]}
    elif args.model in ['extra', 'random']:
        model_params = {"clf__max_depth": [None, 5],
                        "clf__min_samples_leaf": [1, 2]}

    return pipeline, {**selector_params, **model_params}


def _save_shap_values(args, best_model, X_train, X_test, fold, feature_names):

    root = f"{args.results_root}_shap/{args.feature_set_name}/{args.model}/{args.selector}/fold_{fold + 1}"
    os.makedirs(root, exist_ok=True)
    explainer = shap.KernelExplainer(best_model.predict_proba, X_train)
    shap_values = explainer.shap_values(X_test)
    with open(os.path.join(root, 'shap_values.pkl'), 'wb') as f:
        pickle.dump(shap_values, f)
    with open(os.path.join(root, 'X_test.pkl'), 'wb') as f:
        pickle.dump(pd.DataFrame(X_test, columns=feature_names), f)
