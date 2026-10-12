import os
import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from seed import init_seed
from dataset import MySlideDataset
from models.abmil import GatedAttention
from models.transmil import TransMIL
from models.multimodal import Multimodal


def load_clinical_data(args, slide_ids, device):
    """Dict slide_id -> (1, n_features) tensor with the selected tabular features."""

    df = pd.read_csv(args.clinical_csv).set_index('slide_id')[args.feature_columns]
    missing = set(slide_ids) - set(df.index)
    if missing:
        raise ValueError(f"Slides without tabular features in {args.clinical_csv}: {sorted(missing)}")
    return {sid: torch.tensor(df.loc[[sid]].values, dtype=torch.float).to(device) for sid in slide_ids}


def build_model(args, n_clinical):

    if args.model == "abmil":
        return GatedAttention()
    if args.model == "transmil":
        return TransMIL()
    if args.model == "multimodal":
        if not args.use_clinical:
            return Multimodal(input_modalities=['WSI'])
        if args.fixed_dim:
            return Multimodal(wsi_inner_dim=64, clinical_dim=n_clinical, clinical_emb_dim=64, input_modalities=['WSI', 'Clinical'], fuse_modality=args.fuse)
        return Multimodal(wsi_inner_dim=n_clinical, clinical_dim=n_clinical, clinical_emb_dim=n_clinical, input_modalities=['WSI', 'Clinical'], fuse_modality=args.fuse)
    raise ValueError(f"Unsupported model: {args.model}")


def evaluate_model(model, val_indices, dataset, device, clinical_data):

    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for idx in val_indices:
            data = dataset[idx]
            features = data['features'].to(device)
            label = data['label'].to(device).view(-1)
            clinical = clinical_data[data['slide_id']] if clinical_data is not None else None

            logits, prob, pred = model(features, clinical)
            loss = criterion(logits, label)

            total_loss += loss.item()
            all_preds.append(pred.item())
            all_labels.append(label.item())

    avg_loss = total_loss / len(val_indices)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, f1

def train_and_validate(model, train_indices, val_indices, dataset, device, clinical_data, epochs, lr, weight_decay):
    """Train for the given epochs and return the weights of the epoch with the best validation F1 (ties broken by loss)."""

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        np.random.shuffle(train_indices)

        for idx in train_indices:
            data = dataset[idx]
            features = data['features'].to(device)
            label = data['label'].to(device).view(-1)
            clinical = clinical_data[data['slide_id']] if clinical_data is not None else None

            optimizer.zero_grad()
            logits, _, _ = model(features, clinical)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

        scheduler.step()

        val_loss, val_f1 = evaluate_model(model, val_indices, dataset, device, clinical_data)

        is_better_f1 = val_f1 > best_val_f1
        is_same_f1_better_loss = (val_f1 == best_val_f1) and (val_loss < best_val_loss)

        if is_better_f1 or is_same_f1_better_loss:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)

    return model, best_val_f1, best_val_loss

def run_dl_pipeline(args):

    results_path = f"{args.results_root}/{args.model}_{args.lr}_{args.wd}.txt"
    # the results file is written only at the end of the LOOCV
    if args.skip_done and os.path.exists(results_path):
        print(f"Skipping {results_path}: already done")
        return
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    init_seed(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    splitter = LeaveOneOut()

    dataset = MySlideDataset(data_path=args.features_dir)
    labels_all = np.array([data['label'].item() for data in dataset.data])
    names_all = [data['slide_id'] for data in dataset.data]

    if args.use_clinical:
        clinical_data = load_clinical_data(args, names_all, device)
        n_clinical = len(args.feature_columns)
    else:
        clinical_data = None
        n_clinical = None

    y_true_global = []
    y_pred_global = []
    misclassified_ids = []

    print(f"LOOCV on {len(dataset)} samples")

    for outer_fold, (trainval_idx, test_idx) in enumerate(splitter.split(labels_all)):

        test_name = names_all[test_idx[0]]
        print(f"Fold {outer_fold+1}/{len(dataset)} - Test sample: {test_name}")

        trainval_labels = labels_all[trainval_idx]

        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)

        best_inner_model = None
        best_inner_f1 = -1.0
        best_inner_loss = float('inf')

        for train_idx_inner_rel, val_idx_inner_rel in inner_cv.split(trainval_idx, trainval_labels):

            train_idx_abs = trainval_idx[train_idx_inner_rel]
            val_idx_abs = trainval_idx[val_idx_inner_rel]

            model = build_model(args, n_clinical).to(device)
            model, val_f1, val_loss = train_and_validate(
                model,
                train_indices=train_idx_abs,
                val_indices=val_idx_abs,
                dataset=dataset,
                device=device,
                clinical_data=clinical_data,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.wd,
            )

            is_better_f1 = val_f1 > best_inner_f1
            is_same_f1_lower_loss = (val_f1 == best_inner_f1) and (val_loss < best_inner_loss)

            if is_better_f1 or is_same_f1_lower_loss:
                best_inner_f1 = val_f1
                best_inner_loss = val_loss
                best_inner_model = model

        best_inner_model.eval()
        with torch.no_grad():
            data_test = dataset[test_idx[0]]
            clinical = clinical_data[test_name] if clinical_data is not None else None
            _, _, pred = best_inner_model(data_test['features'].to(device), clinical)

        real_label = data_test['label'].item()
        y_true_global.append(real_label)
        y_pred_global.append(pred.item())

        if pred.item() != real_label:
            misclassified_ids.append(test_name)
            print(f"  -> WRONG")
        else:
            print(f"  -> CORRECT")

    final_f1 = f1_score(y_true_global, y_pred_global, average='macro')
    cm = confusion_matrix(y_true_global, y_pred_global)

    with open(results_path, "w") as f:
        f.write(f"\n--- RESULTS ---\n")
        if args.use_clinical:
            f.write(f"Clinical Data: {args.feature_columns}\n")
        f.write(f"Macro F1: {final_f1:.4f}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
        f.write(f"Misclassified IDs: {misclassified_ids}\n")

    print(f"F1: {final_f1:.4f}")
