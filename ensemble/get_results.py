import os
import json
import pandas as pd

BASE_DIR = "ensemble/results/exp3_mlp"
OUTPUT_FILE = os.path.join(BASE_DIR, "results.xlsx")

DATASETS = [
    "3planes_raw_features",
    "img_proj_tab",
    "img_raw_features",
    "img_raw_tab",
]

CONFIGS = [
    "all_data",
    "all_features",
    "all_samples",
    "edca_selection",
]

VOTING_METHODS = [
    "mean_voting",
    "max_voting",
]


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def load_metrics(metrics_path):
    with open(metrics_path, "r") as f:
        m = json.load(f)

    tp = m.get("tp", 0)
    fn = m.get("fn", 0)
    fp = m.get("fp", 0)
    tn = m.get("tn", 0)

    npv = safe_div(tn, tn + fn)
    specificity = safe_div(tn, tn + fp)

    return {
        "Balanced Accuracy": m.get("balanced_accuracy"),
        "Precision": m.get("precision"),
        "Recall": m.get("recall"),
        "F1-score": m.get("f1"),
        "Macro F1": m.get("f1_macro"),
        "ROC AUC": m.get("roc_auc"),
        "Matthews corrcoef": m.get("matthews"),
        "NPV": npv,
        "Specificity": specificity,
        "TP (CS)": tp,
        "FN": fn,
        "FP": fp,
        "TN (VD)": tn,
    }


def build_dataset_table(dataset_dir):
    rows = []

    for voting in VOTING_METHODS:
        for config in CONFIGS:
            metrics_path = os.path.join(dataset_dir, config, voting, "metrics.json")

            row = {
                "Voting": voting,
                "Configuration": config,
            }

            if os.path.isfile(metrics_path):
                row.update(load_metrics(metrics_path))
            else:
                row.update({
                    "Balanced Accuracy": None,
                    "Precision": None,
                    "Recall": None,
                    "F1-score": None,
                    "Macro F1": None,
                    "ROC AUC": None,
                    "Matthews corrcoef": None,
                    "NPV": None,
                    "Specificity": None,
                    "TP (CS)": None,
                    "FN": None,
                    "FP": None,
                    "TN (VD)": None,
                })

            rows.append(row)

    df = pd.DataFrame(rows)

    column_order = [
        "Voting",
        "Configuration",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "Macro F1",
        "ROC AUC",
        "Matthews corrcoef",
        "NPV",
        "Specificity",
        "TP (CS)",
        "FN",
        "FP",
        "TN (VD)",
    ]

    df = df[column_order]

    # round numeric metrics to 3 decimals
    metric_cols = [
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "Macro F1",
        "ROC AUC",
        "Matthews corrcoef",
        "NPV",
        "Specificity",
    ]

    df[metric_cols] = df[metric_cols].astype(float).round(3)

    return df   


def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        for dataset in DATASETS:
            dataset_dir = os.path.join(BASE_DIR, dataset)

            if not os.path.isdir(dataset_dir):
                print(f"Skipping missing dataset folder: {dataset_dir}")
                continue

            df = build_dataset_table(dataset_dir)
            df.to_excel(writer, sheet_name=dataset[:31], index=False)

    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()