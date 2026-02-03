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

# ======================================================
# MÉTRICAS (igual às que enviaste)
# ======================================================

def calculate_metrics(y_test, y_pred, y_proba_1):

    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=1)
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
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
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "matthews": matthews,
        "tpr": tpr,
        "tnr": tnr,
        "specificity": specificity,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# ======================================================
# PROCESSAR UM FICHEIRO CSV
# ======================================================

def process_prediction_file(csv_path):

    df = pd.read_csv(csv_path)

    y_test = df["y_test"].values
    y_pred = df["y_pred"].values
    y_proba_1 = df["y_proba_1"].values

    return calculate_metrics(y_test, y_pred, y_proba_1)


# ======================================================
# PROCESSAR TODA A PASTA DE PREDICTIONS
# ======================================================

def analyze_predictions(predictions_dir, output_file):

    files = [f for f in os.listdir(predictions_dir) if f.endswith(".csv")]

    # padrão: edca_all_data_predictions_1.csv
    pattern = re.compile(r"(.*)_predictions_(\d+)\.csv")

    grouped = {}

    for file in files:
        match = pattern.match(file)
        if match:
            kind, fold = match.groups()
            grouped.setdefault(kind, []).append((int(fold), file))

    writer = pd.ExcelWriter(output_file, engine="openpyxl")

    for kind, entries in grouped.items():

        rows = []

        for fold, filename in sorted(entries):
            csv_path = os.path.join(predictions_dir, filename)
            metrics = process_prediction_file(csv_path)
            metrics["cv"] = f"cv{fold}"
            rows.append(metrics)

        df = pd.DataFrame(rows).set_index("cv")

        # adicionar mean e std
        df.loc["mean"] = df.mean()
        df.loc["std"] = df.std()

   
        sheet_name = kind[:31]
        df.to_excel(writer, sheet_name=sheet_name)

    writer.close()
    print(f"✔ Métricas guardadas em: {output_file}")


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    PREDICTIONS_DIR = "../logs/exp2/testing/exp2_MedViT-nopt/exp_2026-01-30 14:35:15.722061/edca/predictions"

    OUTPUT_FILE = "metrics_predictions_3.xlsx"

    analyze_predictions(PREDICTIONS_DIR, OUTPUT_FILE)
