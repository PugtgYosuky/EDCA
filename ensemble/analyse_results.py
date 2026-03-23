import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


base_dir = "results_prospective"
clinical_path = os.path.join(base_dir, "predictions_clinical_PC.csv")
model_path = os.path.join(base_dir, "MedViT2_nopt", "predictions.csv")
out_dir = os.path.join(base_dir, "alignment_plots")
os.makedirs(out_dir, exist_ok=True)


clinical = pd.read_csv(clinical_path)
model = pd.read_csv(model_path)

clinical["Processo"] = clinical["Processo"].astype(str)
model["Processo"] = model["Processo"].astype(str)

clinical = clinical[["Processo", "y_true", "y_pred", "y_proba_1"]].rename(
    columns={"y_true": "y_true", "y_pred": "y_pred_clin", "y_proba_1": "p_clin"}
)

model = model[["Processo", "y_pred", "y_proba_1"]].rename(
    columns={"y_pred": "y_pred_model", "y_proba_1": "p_model"}
)

df = clinical.merge(model, on="Processo", how="inner")


df["p_clin"] = pd.to_numeric(df["p_clin"], errors="coerce").clip(0, 1)
df["p_model"] = pd.to_numeric(df["p_model"], errors="coerce").clip(0, 1)
df = df.dropna(subset=["p_clin", "p_model", "y_true", "y_pred_clin", "y_pred_model"]).copy()



df["clin_correct"] = df["y_pred_clin"] == df["y_true"]
df["model_correct"] = df["y_pred_model"] == df["y_true"]

def group_label(r):
    if r["clin_correct"] and r["model_correct"]:
        return "Both correct"
    if (not r["clin_correct"]) and (not r["model_correct"]):
        return "Both wrong"
    if (not r["clin_correct"]) and r["model_correct"]:
        return "Model only correct"
    return "Clinical only correct"

df["group"] = df.apply(group_label, axis=1)


colors = {
    "Both correct": "tab:green",
    "Both wrong": "tab:red",
    "Model only correct": "tab:blue",
    "Clinical only correct": "tab:orange",
}
group_order = ["Both correct", "Both wrong", "Model only correct", "Clinical only correct"]


# Plot function

def corr_text(data):
    if len(data) < 3:
        return f"N = {len(data)}\nPearson r = NA\nSpearman ρ = NA"
    pr = pearsonr(data["p_clin"], data["p_model"])[0]
    sr = spearmanr(data["p_clin"], data["p_model"])[0]
    return f"N = {len(data)}\nPearson r = {pr:.3f}\nSpearman ρ = {sr:.3f}"

def plot_alignment(df_all, title, filename, focus_group=None, show_context=True):
   
    plt.figure(figsize=(7, 7))

    if focus_group is None:
        # Full colored plot (all data)
        for g in group_order:
            sub = df_all[df_all["group"] == g]
            plt.scatter(
                sub["p_clin"], sub["p_model"],
                s=30, alpha=0.75,
                c=colors[g],
                label=f"{g} (n={len(sub)})",
                edgecolors="none"
            )
        corr_data = df_all

    else:
        # Focus plot: highlight one group
        focus = df_all[df_all["group"] == focus_group]
        others = df_all[df_all["group"] != focus_group]

        if show_context and len(others) > 0:
            plt.scatter(
                others["p_clin"], others["p_model"],
                s=25, alpha=0.20,
                c="lightgray",
                label=f"Other groups (n={len(others)})",
                edgecolors="none"
            )

        plt.scatter(
            focus["p_clin"], focus["p_model"],
            s=35, alpha=0.85,
            c=colors[focus_group],
            label=f"{focus_group} (n={len(focus)})",
            edgecolors="none"
        )
        corr_data = focus

    # diagonal reference
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Clinical practice probability (p_clin)")
    plt.ylabel("Model probability (p_model)")
    plt.title(title)
    plt.grid(alpha=0.3)

    # correlations on the subset being “evaluated” (all or focus)
    plt.text(
        0.02, 0.98, corr_text(corr_data),
        transform=plt.gca().transAxes,
        ha="left", va="top"
    )

    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=200)
    plt.close()

    # also print correlations
    print(title)
    print(corr_text(corr_data))
    print("-" * 40)


plot_alignment(df, "All samples (colored by correctness)", "01_all_samples_colored.png", focus_group=None)

plot_alignment(df, "Both correct (highlighted)", "02_both_correct.png", focus_group="Both correct", show_context=True)
plot_alignment(df, "Both wrong (highlighted)", "03_both_wrong.png", focus_group="Both wrong", show_context=True)
plot_alignment(df, "Model only correct (highlighted)", "04_model_only_correct.png", focus_group="Model only correct", show_context=True)
plot_alignment(df, "Clinical only correct (highlighted)", "05_clinical_only_correct.png", focus_group="Clinical only correct", show_context=True)

print(f"\nSaved plots to: {out_dir}")
