import os
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations
from statsmodels.stats.multitest import multipletests

base_path = "../results"

configs = {
    "img_raw_tab": "config_i",
    "img_proj_tab": "config_ii",
    "img_raw_features": "config_iii",
    "3planes_raw_features": "config_iv"
}

training_modes_main = {
    "all_data": "edca_all_data_predictions_1",
    "all_features": "edca_all_features_predictions_1",
    "all_samples": "edca_all_samples_predictions_1",
    "edca": "edca_predictions_1"}

training_modes_extend = {
    **training_modes_main,
    "selected_image": "edca_selected_image_features_pr",
    "selected_tabular": "edca_selected_tabular_features_",
    "all_tabular": "edca_all_tabular_features_predi",
    "all_image": "edca_all_image_features_predict"
}

metrics = ["roc_auc", "f1", "precision", "tp", "fp","tn", "fn"]

results = []


# ======================
# LOAD DATA
# ======================
def load_data():
    data = {}

    for cfg_folder, cfg_name in configs.items():
        file_path = os.path.join(
            base_path,
            cfg_folder,
            "exp3_mlp",
            "metrics_predictions_MedViT2_nopt.xlsx"
        )

        xls = pd.ExcelFile(file_path)
        data[cfg_name] = {}

        available_sheets = xls.sheet_names

        for mode_name, sheet_pattern in training_modes_extend.items():

            matched_sheet = None
            for s in available_sheets:
                if sheet_pattern in s:
                    matched_sheet = s
                    break

            if matched_sheet is None:
                print(f"[WARNING] Sheet not found for {mode_name} in {cfg_name}")
                continue

            df = pd.read_excel(xls, sheet_name=matched_sheet)
            df = df.iloc[:30].reset_index(drop=True)

            data[cfg_name][mode_name] = df

    return data

# ======================
# WILCOXON + BONFERRONI
# ======================
def wilcoxon_posthoc(combined_df, context_info):
    pairs = list(combinations(combined_df.columns, 2))
    p_values = []

    # compute raw p-values
    for a, b in pairs:
        stat, p = wilcoxon(
            combined_df[a],
            combined_df[b],
            zero_method='wilcox',
            alternative='two-sided'
        )
        p_values.append(p)

    # Bonferroni correction
    _, pvals_corrected, _, _ = multipletests(
        p_values, method='bonferroni'
    )

    # store results
    for i, (a, b) in enumerate(pairs):
        results.append({
            "test_type": context_info["test_type"],
            "context": context_info["context"],
            "metric": context_info["metric"],
            "comparison": f"{a} vs {b}",
            "bonf_p": pvals_corrected[i],
            "sig_95": pvals_corrected[i] < 0.05,
            "sig_99": pvals_corrected[i] < 0.01
        })


# ======================
# BETWEEN CONFIGS
# ======================
def test_between_configs(data, metric, mode):
    dfs = []

    for cfg in configs.values():
        dfs.append(data[cfg][mode][metric])

    combined = pd.concat(dfs, axis=1)
    combined.columns = list(configs.values())
    combined = combined.dropna()

    if combined.shape[0] < 2:
        return

    stat, p = friedmanchisquare(*[combined[c] for c in combined.columns])

    if p < 0.05:  # trigger posthoc
        context_info = {
            "test_type": "between_configs",
            "context": mode,
            "metric": metric
        }

        wilcoxon_posthoc(combined, context_info)


# ======================
# WITHIN CONFIG (CONFIG IV)
# ======================
def test_within_config(data, config_name, metric):
    dfs = []
    valid_modes = []

    for mode in training_modes_extend.keys():
        if mode in data[config_name]:
            dfs.append(data[config_name][mode][metric])
            valid_modes.append(mode)

    combined = pd.concat(dfs, axis=1)
    combined.columns = valid_modes
    combined = combined.dropna()

    if combined.shape[0] < 2:
        return

    stat, p = friedmanchisquare(*[combined[c] for c in combined.columns])

    if p < 0.05:
        context_info = {
            "test_type": "within_config",
            "context": config_name,
            "metric": metric
        }

        wilcoxon_posthoc(combined, context_info)


# ======================
# RUN ALL TESTS
# ======================
data = load_data()

# TEST 1: between configs
for mode in training_modes_main.keys():
    for metric in metrics:
        test_between_configs(data, metric, mode)

# TEST 2: within config IV only
for metric in metrics:
    test_within_config(data, "config_iv", metric)


# ======================
# SAVE RESULTS
# ======================
results_df = pd.DataFrame(results)

results_df.to_csv("stat_significance.csv", index=False)
results_df.to_excel("stat_significance.xlsx", index=False)

print("Saved: stat_significance.csv / .xlsx")