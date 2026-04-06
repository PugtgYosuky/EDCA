import os
import re
import numpy as np
import pandas as pd

from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score
)

def calculate_metrics(y_test, y_pred, y_proba_1):
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=1)
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    matthews = matthews_corrcoef(y_test, y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred, labels=[1, 0])
    roc_auc = roc_auc_score(y_test, y_proba_1)

    tp, fn, fp, tn = conf_matrix.ravel()

    tpr = tp / (tp + fp) if (tp + fp) > 0 else 0
    tnr = tn / (tn + fn) if (tn + fn) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "matthews": matthews,
        "tnr": tnr,
        "specificity": specificity,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
    }

def process_prediction_file(csv_path):
    df = pd.read_csv(csv_path)

   
    if "y_test" in df.columns:
        y_test = df["y_test"].values
    elif "y_true" in df.columns:
        y_test = df["y_true"].values
    else:
        raise ValueError(f"Missing y_test/y_true in {csv_path}")

    if "y_pred" not in df.columns:
        raise ValueError(f"Missing y_pred in {csv_path}")
    y_pred = df["y_pred"].values

    if "y_proba_1" in df.columns:
        y_proba_1 = df["y_proba_1"].values
    elif "prob_1" in df.columns:
        y_proba_1 = df["prob_1"].values
    else:
        raise ValueError(f"Missing y_proba_1 (or prob_1) in {csv_path}")

    return calculate_metrics(y_test, y_pred, y_proba_1)

def find_prediction_csvs(experiments_root):
    """
    Finds prediction CSVs inside each exp_* folder, supporting both:
      exp_*/edca/predictions/*.csv
      exp_*/edca/edca_fold*/predictions/*.csv
    Returns list of (exp_name, csv_path).
    """
    out = []
    for exp in sorted(os.listdir(experiments_root)):
        exp_dir = os.path.join(experiments_root, exp)
        if not os.path.isdir(exp_dir) or not exp.startswith("exp_"):
            continue

        # common locations
        candidates = [
            os.path.join(exp_dir, "edca", "predictions"),
        ]

        # also allow fold folders
        edca_dir = os.path.join(exp_dir, "edca")
        if os.path.isdir(edca_dir):
            for sub in os.listdir(edca_dir):
                if sub.startswith("edca_fold"):
                    candidates.append(os.path.join(edca_dir, sub, "predictions"))

        for pred_dir in candidates:
            if not os.path.isdir(pred_dir):
                continue
            for f in os.listdir(pred_dir):
                if f.endswith(".csv"):
                    out.append((exp, os.path.join(pred_dir, f)))
    return out

def analyze_predictions(experiments_root, output_file):
    found = find_prediction_csvs(experiments_root)
    if not found:
        raise FileNotFoundError(f"No prediction CSVs found under: {experiments_root}")

    # group by filename "kind" (everything before .csv), so different prediction files go to different sheets
    grouped = {}
    for exp_name, csv_path in found:
        fname = os.path.basename(csv_path)
        kind = fname.replace(".csv", "")
        grouped.setdefault(kind, []).append((exp_name, csv_path))

    
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    writer = pd.ExcelWriter(output_file, engine="openpyxl")


    for kind, entries in grouped.items():
        rows = []
        for exp_name, csv_path in entries:
            metrics = process_prediction_file(csv_path)
            metrics["exp"] = exp_name
            rows.append(metrics)

        df = pd.DataFrame(rows).set_index("exp")

        # mean/std across runs (exp folders)
        df.loc["mean"] = df.mean(numeric_only=True)
        df.loc["std"] = df.std(numeric_only=True)

        sheet_name = kind[:31]
        df.round(3).to_excel(writer, sheet_name=sheet_name)

    writer.close()
    print(f"✔ Métricas guardadas em: {output_file}")

if __name__ == "__main__":
  
    EXPERIMENTS_ROOT = "../logs/MedViT2-nopt/3planes_raw_features/exp4"
    OUTPUT_FILE = "../results/3planes_raw_features/exp4/metrics_predictions_MedViT2_nopt.xlsx"

    analyze_predictions(EXPERIMENTS_ROOT, OUTPUT_FILE)
