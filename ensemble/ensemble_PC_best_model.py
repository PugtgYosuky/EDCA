import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, recall_score, balanced_accuracy_score, f1_score,
    precision_score, matthews_corrcoef, confusion_matrix
)


model_preds_path = "../logs/MedViT2-nopt/3planes_raw_features/exp4/exp_2026-03-24 09:09:42.924309/edca/predictions/edca_predictions_1.csv"
processo_path = "../data/datasets/MedViT2-nopt/all_data/prospective/PT_external_dataset_processed.csv"
clinical_path = "../data/datasets/Base_dados_prospetiva.xlsx"

output_root = Path("ensemble/results/exp4/3planes_raw_features/edca_selection")
output_root.mkdir(parents=True, exist_ok=True)

clinical_processo_col = "Processo"
clinical_practice_value = "PC"   

voting_methods = ["mean_voting", "max_voting"]


def compute_metrics(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tp, fn, fp, tn = cm.ravel()

    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=1),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "matthews": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
    }
    return metrics


def normalize_0_10_to_0_1(x):
    return pd.to_numeric(x, errors="coerce") / 10.0


# load model predictions

df_model = pd.read_csv(model_preds_path)


if "y_true" not in df_model.columns and "y_test" in df_model.columns:
    df_model = df_model.rename(columns={"y_test": "y_true"})

# add processo to predictions file 
df_proc = pd.read_csv(processo_path)

if "Processo" not in df_proc.columns:
    raise ValueError("Column 'Processo' not found in PT_external_dataset_processed.csv")

if len(df_model) != len(df_proc):
    raise ValueError(
        f"Different number of rows: predictions={len(df_model)} vs processo file={len(df_proc)}"
    )

df_model = df_model.copy()
df_model["Processo"] = df_proc["Processo"].values

# load clinical practice
df_clinical = pd.read_excel(clinical_path)

if clinical_processo_col not in df_clinical.columns:
    raise ValueError(f"Column '{clinical_processo_col}' not found in clinical file")

if clinical_practice_value not in df_clinical.columns:
    raise ValueError(f"Column '{clinical_practice_value}' not found in clinical file")

df_clinical = df_clinical[[clinical_processo_col, clinical_practice_value]].copy()
df_clinical = df_clinical.rename(columns={
    clinical_processo_col: "Processo",
    clinical_practice_value: "clinical_practice_raw"
})

df_clinical["Processo"] = df_clinical["Processo"].astype(str)
df_clinical["p_human"] = normalize_0_10_to_0_1(df_clinical["clinical_practice_raw"])
df_clinical["p_human"] = df_clinical["p_human"].clip(0, 1)

# merge both predictions
df_model["Processo"] = df_model["Processo"].astype(str)

df_merged = pd.merge(
    df_model,
    df_clinical[["Processo", "clinical_practice_raw", "p_human"]],
    on="Processo",
    how="inner"
)

print(f"Merged rows: {len(df_merged)}")


if "y_pred" not in df_merged.columns:
    raise ValueError("Column 'y_pred' not found in model predictions")

if "y_proba_1" not in df_merged.columns:
    raise ValueError("Column 'y_proba_1' not found in model predictions")

if "y_true" not in df_merged.columns:
    raise ValueError("Column 'y_true' or 'y_test' not found in model predictions")

df_merged["p_model"] = pd.to_numeric(df_merged["y_proba_1"], errors="coerce")
df_merged["y_pred_model"] = pd.to_numeric(df_merged["y_pred"], errors="coerce").astype(int)
df_merged["y_true"] = pd.to_numeric(df_merged["y_true"], errors="coerce").astype(int)
df_merged["y_pred_human"] = (df_merged["p_human"] >= 0.5).astype(int)

# save merged base file
df_merged.to_csv(output_root / "merged_model_cp.csv", index=False)

#  ENSEMBLE 
for method in voting_methods:
    y_true = df_merged["y_true"].values
    p_model = df_merged["p_model"].values
    p_human = df_merged["p_human"].values

    pred_model = df_merged["y_pred_model"].values
    pred_human = df_merged["y_pred_human"].values

    if method == "mean_voting":
        p_ensemble = np.mean([p_model, p_human], axis=0)
        pred_ensemble = (p_ensemble >= 0.5).astype(int)

    elif method == "max_voting":
        preds_stack = np.stack([pred_model, pred_human], axis=1)
        pred_ensemble = np.array([np.bincount(row).argmax() for row in preds_stack])
        p_ensemble = np.mean([p_model, p_human], axis=0)

    metrics = compute_metrics(y_true, pred_ensemble, p_ensemble)

    save_dir = output_root / method
    save_dir.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame({
        "Processo": df_merged["Processo"],
        "y_true": y_true,
        "y_pred_model": pred_model,
        "y_proba_model": p_model,
        "clinical_practice_raw": df_merged["clinical_practice_raw"],
        "y_pred_human": pred_human,
        "y_proba_human": p_human,
        "y_pred_ensemble": pred_ensemble,
        "y_proba_ensemble": p_ensemble,
    })
    df_out.to_csv(save_dir / "predictions_ensemble.csv", index=False)

    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"{method}: {metrics}")
    print(f"Saved in {save_dir}")

print("Done")