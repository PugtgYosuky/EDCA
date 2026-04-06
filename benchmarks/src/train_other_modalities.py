import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from utils import append_metrics, save_predictions
from edca.utils import class_distribution_distance
from edca.estimator import PipelineEstimator


DATASET_KINDS = {
    "img_raw_tab",
    "img_proj_tab",
    "img_raw_features",
    "3planes_raw_features",
}


def infer_dataset_kind(config, run_dir):
    text = " | ".join([
        str(config.get("save_path", "")),
        str(config.get("train_dataset", "")),
        str(run_dir),
    ])
    for kind in DATASET_KINDS:
        if kind in text:
            return kind
    raise ValueError(f"Could not infer dataset kind from: {text}")


def split_feature_groups(columns, dataset_kind):
    cols = list(columns)

    if dataset_kind in {"img_raw_tab", "img_proj_tab"}:
        img_cols = [c for c in cols if c.startswith("img_emb")]
        tab_cols = [c for c in cols if c.startswith("tab_emb")]

    elif dataset_kind == "img_raw_features":
        img_cols = [c for c in cols if c.startswith("img_emb")]
        tab_cols = [c for c in cols if c not in img_cols]

    elif dataset_kind == "3planes_raw_features":
        img_prefixes = ("image_femur_emb", "image_abdomen_emb", "image_head_emb")
        img_cols = [c for c in cols if c.startswith(img_prefixes)]
        tab_cols = [c for c in cols if c not in img_cols]

    else:
        raise ValueError(f"Unknown dataset kind: {dataset_kind}")

    return tab_cols, img_cols


def load_run_files(run_dir):
    run_dir = Path(run_dir)

    with open(run_dir / "config.json", "r") as f:
        config = json.load(f)

    with open(run_dir / "results.json", "r") as f:
        results = json.load(f)

    with open(run_dir / "edca" / "edca_fold1" / "pipeline_config.json", "r") as f:
        pipeline_config = json.load(f)

    return config, results, pipeline_config


def load_train_test_data(config, results):
    train_path = results["train_path"]
    test_path = results["test_path"]

    label_col = config.get("label_col", "Class")
    drop_cols = config.get("drop_cols", [])

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    y_train = df_train.pop(label_col)
    y_test = df_test.pop(label_col)

    for c in drop_cols:
        if c in df_train.columns:
            df_train = df_train.drop(columns=[c])
        if c in df_test.columns:
            df_test = df_test.drop(columns=[c])

    df_test = df_test[df_train.columns]

    return df_train, y_train, df_test, y_test


def load_selected_feature_names(run_dir, label_col):
    best_data_path = Path(run_dir) / "edca" / "edca_fold1" / "best_data.csv"
    best_df = pd.read_csv(best_data_path)
    return [c for c in best_df.columns if c != label_col]


def build_feature_sets(X_train_columns, selected_feature_names, dataset_kind):
    tab_all, img_all = split_feature_groups(X_train_columns, dataset_kind)
    selected_set = set(selected_feature_names)

    tab_selected = [c for c in tab_all if c in selected_set]
    img_selected = [c for c in img_all if c in selected_set]

    return {
        "edca_all_tabular_features": tab_all,
        "edca_all_image_features": img_all,
        "edca_selected_tabular_features": tab_selected,
        "edca_selected_image_features": img_selected,
    }


def retrain_best_individual_from_subset(
    name,
    best_individual,
    pipeline_config,
    fairness_params,
    X_train,
    y_train,
    X_test,
    y_test,
    save_name,
):
    individual = deepcopy(best_individual)

    # remove EDCA data reduction genes because the dataframe subset
    # we pass in already defines the data used for retraining
    for dr in ["sample", "features"]:
        if dr in individual:
            individual.pop(dr)

    pipeline_estimator = PipelineEstimator(
        individual_config=individual,
        pipeline_config=pipeline_config,
        individual_id=name,
        fairness_params=fairness_params,
    )

    pipeline_estimator.fit(X_train, y_train)

    preds = pipeline_estimator.predict(X_test)
    proba_preds = pipeline_estimator.predict_proba(X_test)

    if fairness_params:
        sensitive_cols = fairness_params.get("sensitive_attributes", [])
        sensitive_data = X_test[sensitive_cols] if sensitive_cols else pd.DataFrame()
    else:
        sensitive_data = pd.DataFrame()

    framework_results = {}
    append_metrics(
        results=framework_results,
        y_test=y_test,
        preds=preds,
        proba_preds=proba_preds,
        final_data_size=X_train.shape,
        original_data_size=X_train.shape,
        cdd=class_distribution_distance(
            np.array(y_train.value_counts(normalize=True)),
            y_train.nunique()
        ),
        class_proportions=y_train.value_counts().to_dict(),
        sensitive_attributes=sensitive_data,
        fairness_params=fairness_params,
    )
    framework_results["best_pipeline_time"] = pipeline_estimator.train_time

    save_predictions(
        filename=str(save_name),
        y_test=y_test,
        y_preds=preds,
        y_preds_proba=proba_preds,
        fairness_sensitive_attributes=sensitive_data,
    )

    return framework_results


def process_run(run_dir, overwrite=False):
    run_dir = Path(run_dir)
    print(f"\nProcessing {run_dir}")

    config, results, pipeline_config = load_run_files(run_dir)
    X_train, y_train, X_test, y_test = load_train_test_data(config, results)

    dataset_kind = infer_dataset_kind(config, run_dir)
    label_col = config.get("label_col", "Class")
    fairness_params = config.get("fairness_params", {}) or {}

    best_individual = results["run_info"]["edca"]["best"]
    selected_feature_names = load_selected_feature_names(run_dir, label_col)

    feature_sets = build_feature_sets(
        X_train_columns=X_train.columns,
        selected_feature_names=selected_feature_names,
        dataset_kind=dataset_kind,
    )

    print("Feature groups:")
    for k, v in feature_sets.items():
        print(f"  {k}: {len(v)}")

    pred_dir = run_dir / "edca" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    new_results = {}

    for result_name, cols in feature_sets.items():
        pred_file = pred_dir / f"{result_name}_predictions_1.csv"

        if result_name in results["run_info"] and not overwrite:
            print(f"  Skipping {result_name}: already in results.json")
            continue

        if pred_file.exists() and not overwrite:
            print(f"  Skipping {result_name}: prediction file already exists")
            continue

        if len(cols) == 0:
            print(f"  Skipping {result_name}: 0 columns")
            continue

        missing = [c for c in cols if c not in X_train.columns]
        if missing:
            print(f"  Skipping {result_name}: missing columns, first few: {missing[:5]}")
            continue

        print(f"  Training {result_name} with {len(cols)} features")

        Xtr = X_train[cols].copy()
        Xte = X_test[cols].copy()

        out = retrain_best_individual_from_subset(
            name=result_name,
            best_individual=best_individual,
            pipeline_config=pipeline_config,
            fairness_params=fairness_params,
            X_train=Xtr,
            y_train=y_train,
            X_test=Xte,
            y_test=y_test,
            save_name=pred_file,
        )

        out["num_features_used"] = len(cols)
        out["feature_names_used"] = cols
        new_results[result_name] = out



def process_experiment(experiment_dir, overwrite=False):
    experiment_dir = Path(experiment_dir)

    run_dirs = sorted(
        p for p in experiment_dir.iterdir()
        if p.is_dir() and p.name.startswith("exp_")
    )

    print(f"Found {len(run_dirs)} runs in {experiment_dir}")

    for run_dir in run_dirs:
        try:
            process_run(run_dir, overwrite=overwrite)
        except Exception as e:
            print(f"Error in {run_dir}: {e}")


if __name__ == "__main__":
    experiment_dir = "logs/MedViT2-nopt/3planes_raw_features/exp4"
    process_experiment(experiment_dir, overwrite=False)