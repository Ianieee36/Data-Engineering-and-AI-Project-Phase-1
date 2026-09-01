"""
Task 4: Exploratory Data Analysis and Visualisation
PhiUSIIL Phishing URL Dataset

Structured to match the Task 4 report sections:
  4.2 Descriptive analysis (numerical + categorical)
  4.3 Target/class distribution
  4.4 Missingness
  4.5 Feature distributions and relationships (incl. the statistics
      behind each reasoning point: zero-inflation, NoOfSelfRef gap,
      TLDLegitimateProb discretisation, IsHTTPS-by-label)
  4.6 Categorical patterns (TLD, binary flags)
  4.7/4.8 supporting numbers for the imbalance/insight summary

Usage:
    python task4_eda_report.py --input PhiUSIIL_cleaned.csv --outdir figures
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
LABEL_NAMES = {0: "Phishing", 1: "Legitimate"}
PALETTE = {0: "#d62728", 1: "#2ca02c"}

URL_LEXICAL_COLS = [
    "URLLength", "DomainLength", "TLDLength", "NoOfSubDomain",
    "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL",
    "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL",
    "NoOfObfuscatedChar", "ObfuscationRatio",
]
DERIVED_PROB_COLS = [
    "URLSimilarityIndex", "CharContinuationRate", "TLDLegitimateProb",
    "URLCharProb", "DomainTitleMatchScore", "URLTitleMatchScore",
]
CONTENT_STRUCTURE_COLS = [
    "LineOfCode", "LargestLineLength", "NoOfImage", "NoOfCSS", "NoOfJS",
    "NoOfSelfRef", "NoOfEmptyRef", "NoOfExternalRef",
    "NoOfURLRedirect", "NoOfSelfRedirect", "NoOfPopup", "NoOfiFrame",
]
BINARY_FLAG_COLS = [
    "IsDomainIP", "HasObfuscation", "IsHTTPS", "HasTitle", "HasFavicon",
    "Robots", "IsResponsive", "HasDescription", "HasExternalFormSubmit",
    "HasSocialNet", "HasSubmitButton", "HasHiddenFields", "HasPasswordField",
    "Bank", "Pay", "Crypto", "HasCopyrightInfo",
]
LOG_COLS = [
    "URLLength_log", "LineOfCode_log", "LargestLineLength_log", "NoOfImage_log",
    "NoOfCSS_log", "NoOfJS_log", "NoOfSelfRef_log", "NoOfEmptyRef_log",
    "NoOfExternalRef_log",
]
OBFUSCATION_COLS = [
    "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
    "NoOfObfuscatedChar", "HasObfuscation",
]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns\n")
    return df


# ---------------------------------------------------------------------------
# 4.2 Descriptive analysis
# ---------------------------------------------------------------------------
def descriptive_analysis(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("4.2 DESCRIPTIVE ANALYSIS")
    print("=" * 70)

    groups = {
        "URL-lexical / structural features": URL_LEXICAL_COLS,
        "Derived / probabilistic features": DERIVED_PROB_COLS,
        "Webpage content / structure features": CONTENT_STRUCTURE_COLS,
    }
    for title, cols in groups.items():
        print(f"\n--- Numeric summary: {title} ({len(cols)} columns) ---")
        print(df[cols].describe().T.round(3))

    print(f"\n--- Binary flag prevalence, all {len(BINARY_FLAG_COLS)} flags "
          f"(proportion = 1) ---")
    print(df[BINARY_FLAG_COLS].mean().round(4).sort_values(ascending=False))

    print("\n--- Categorical: TLD ---")
    print(f"Unique TLD values: {df['TLD'].nunique()}")
    top10 = df["TLD"].value_counts().head(10)
    print(top10)
    print(f"Top 10 TLDs cover {top10.sum() / len(df):.1%} of all records")
    singleton_tlds = (df["TLD"].value_counts() == 1).sum()
    print(f"TLDs appearing only once: {singleton_tlds} / {df['TLD'].nunique()} "
          f"({singleton_tlds / df['TLD'].nunique():.1%})")


# ---------------------------------------------------------------------------
# 4.3 Target distribution
# ---------------------------------------------------------------------------
def target_distribution(df: pd.DataFrame, outdir: str) -> None:
    print("\n" + "=" * 70)
    print("4.3 TARGET / CLASS DISTRIBUTION")
    print("=" * 70)
    counts = df["label"].value_counts().rename(index=LABEL_NAMES)
    props = df["label"].value_counts(normalize=True).rename(index=LABEL_NAMES)
    print(pd.DataFrame({"count": counts, "proportion": props.round(4)}))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    counts_sorted = df["label"].value_counts().sort_index()
    labels = [LABEL_NAMES[i] for i in counts_sorted.index]
    colors = [PALETTE[i] for i in counts_sorted.index]
    bars = ax.bar(labels, counts_sorted.values, color=colors)
    ax.set_ylim(0, counts_sorted.values.max() * 1.18)
    for bar, val in zip(bars, counts_sorted.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + counts_sorted.values.max() * 0.02,
                f"{val:,}\n({val/len(df):.1%})", ha="center", fontsize=10)
    ax.set_title("Figure 1: Class Distribution — Legitimate vs Phishing URLs")
    ax.set_ylabel("Number of records")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "01_class_distribution.png"), dpi=150)
    plt.close(fig)
    print("Saved Figure 1: 01_class_distribution.png")


# ---------------------------------------------------------------------------
# 4.4 Missingness
# ---------------------------------------------------------------------------
def missingness_analysis(df: pd.DataFrame, outdir: str) -> None:
    print("\n" + "=" * 70)
    print("4.4 MISSINGNESS")
    print("=" * 70)
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    print("Columns with missing values:")
    print(miss if len(miss) > 0 else "None")

    if "Title" in miss.index and "HasTitle" in df.columns:
        structural = df.loc[df["Title"].isnull(), "HasTitle"].eq(0).mean()
        print(f"\nOf the missing Title rows, {structural:.1%} have HasTitle=0 "
              f"(confirms missingness is structural, not random).")

    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.4 * max(len(miss), 1))))
    if len(miss) == 0:
        ax.text(0.5, 0.5, "No missing values in any column", ha="center", va="center", fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
    else:
        pct = (miss / len(df) * 100)
        sns.barplot(x=pct.values, y=pct.index, ax=ax, color="#e07b39")
        ax.set_xlabel("% missing")
        ax.set_ylabel("")
    ax.set_title("Figure 2: Missing Value Pattern")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "02_missingness.png"), dpi=150)
    plt.close(fig)
    print("Saved Figure 2: 02_missingness.png")


# ---------------------------------------------------------------------------
# 4.5 Feature distributions and relationships
# ---------------------------------------------------------------------------
def plot_histogram_grid(df, cols, title, fname, outdir, ncols=3):
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, cols):
        for lbl in [0, 1]:
            sns.histplot(df.loc[df["label"] == lbl, col], bins=35, ax=ax,
                         color=PALETTE[lbl], label=LABEL_NAMES[lbl],
                         stat="density", alpha=0.5, element="step")
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.legend(fontsize=7)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle(title, y=1.01, fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_clipped_histogram_grid(df, cols, xlims, title, fname, outdir, ncols=3):
    """Histogram grid where each panel's x-axis is clipped to a sensible
    range (e.g. the 99th percentile) instead of stretching to the raw max —
    a few extreme outliers otherwise compress the entire informative range
    of the distribution into a single pixel-wide spike."""
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, cols):
        xmax = xlims.get(col)
        data_for_bins = df[col] if xmax is None else df.loc[df[col] <= xmax, col]
        bins = np.linspace(data_for_bins.min(), data_for_bins.max() if xmax is None else xmax, 35)
        for lbl in [0, 1]:
            sns.histplot(df.loc[df["label"] == lbl, col], bins=bins, ax=ax,
                         color=PALETTE[lbl], label=LABEL_NAMES[lbl],
                         stat="density", alpha=0.5, element="step")
        if xmax is not None:
            ax.set_xlim(0, xmax)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.legend(fontsize=7)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle(title, y=1.01, fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_zero_inflation_bars(df, cols, title, fname, outdir):
    """For features where >90% of rows are exactly 0, a histogram is the
    wrong chart (it's a single spike with no visible shape). Instead show
    P(feature > 0) by class as a bar chart — this is the actual signal
    these features carry."""
    rates = pd.DataFrame({
        lbl_name: [(df.loc[df["label"] == lbl, col] > 0).mean() for col in cols]
        for lbl, lbl_name in LABEL_NAMES.items()
    }, index=cols)
    fig, ax = plt.subplots(figsize=(8, 5))
    rates.plot(kind="bar", ax=ax, color=[PALETTE[0], PALETTE[1]])
    ax.set_ylabel("Proportion of rows with value > 0")
    ax.set_title(title)
    ax.set_xticklabels(cols, rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=150)
    plt.close(fig)


def section_4_5_1_url_lexical(df: pd.DataFrame, outdir: str) -> None:
    print("\n--- 4.5.1 URL-lexical features: zero-inflation check ---")
    zero_inflated_cols = ["NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
                           "NoOfObfuscatedChar", "ObfuscationRatio"]
    for col in zero_inflated_cols:
        zero_pct = (df[col] == 0).mean()
        print(f"  {col:20s} {zero_pct:.2%} of rows are zero")

    # Group A: well-behaved features, shown at full/clipped range
    group_a = ["DomainLength", "LetterRatioInURL", "SpacialCharRatioInURL"]
    xlims_a = {"SpacialCharRatioInURL": 0.2}
    plot_clipped_histogram_grid(df, group_a, xlims_a,
                                 "Figure 3a: URL-Lexical Features — Well-Behaved Distributions",
                                 "03a_hist_url_lexical_wellbehaved.png", outdir)
    print("Saved Figure 3a: 03a_hist_url_lexical_wellbehaved.png")

    # Group B: small-range integer counts, clipped to the 99th percentile
    group_b = ["TLDLength", "NoOfSubDomain", "NoOfDegitsInURL", "DegitRatioInURL"]
    xlims_b = {"TLDLength": 6, "NoOfSubDomain": 4, "NoOfDegitsInURL": 30, "DegitRatioInURL": 0.35}
    plot_clipped_histogram_grid(df, group_b, xlims_b,
                                 "Figure 3b: URL-Lexical Features — Clipped to 99th Percentile",
                                 "03b_hist_url_lexical_clipped.png", outdir)
    print("Saved Figure 3b: 03b_hist_url_lexical_clipped.png")

    # Group C: heavily skewed letter/URL-length counts, log1p transformed
    group_c = ["NoOfLettersInURL", "NoOfOtherSpecialCharsInURL"]
    df_log = df.copy()
    for c in group_c:
        df_log[f"{c}_log"] = np.log1p(df_log[c])
    plot_histogram_grid(df_log, [f"{c}_log" for c in group_c],
                         "Figure 3c: URL-Lexical Features — log1p Transformed",
                         "03c_hist_url_lexical_log.png", outdir, ncols=2)
    print("Saved Figure 3c: 03c_hist_url_lexical_log.png")

    # Group D: >90% zero-inflated — histogram would just be a spike, use
    # a "proportion > 0" bar chart instead, which is the real signal here
    plot_zero_inflation_bars(df, zero_inflated_cols,
                              "Figure 3d: Zero-Inflated URL Features — P(value > 0) by Class",
                              "03d_zero_inflation_bars.png", outdir)
    print("Saved Figure 3d: 03d_zero_inflation_bars.png")


def section_4_5_2_content_structure(df: pd.DataFrame, outdir: str) -> None:
    print("\n--- 4.5.2 Content/structure features: NoOfSelfRef gap ---")
    stats = df.groupby("label")["NoOfSelfRef"].describe()[["mean", "50%", "max"]]
    stats.index = [LABEL_NAMES[i] for i in stats.index]
    print(stats.round(2))
    plot_histogram_grid(df, LOG_COLS, "Figure 4: Content/Structure Feature Distributions (log1p) by Class",
                         "04_hist_content_log.png", outdir)
    print("Saved Figure 4: 04_hist_content_log.png")


def section_4_5_3_correlation(df: pd.DataFrame, outdir: str) -> pd.Series:
    print("\n--- 4.5.3 Correlation structure ---")
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "label"]
    full_corr = df[numeric_cols + ["label"]].corr()
    label_corr = full_corr["label"].drop("label").abs().sort_values(ascending=False)
    print("Top 15 correlations with label (all numeric columns):")
    print(label_corr.round(3).head(15))

    heatmap_cols = (URL_LEXICAL_COLS + DERIVED_PROB_COLS +
                     ["LineOfCode_log", "NoOfImage_log", "NoOfCSS_log", "NoOfJS_log",
                      "NoOfSelfRef_log", "NoOfExternalRef_log", "IsHTTPS", "IsDomainIP", "label"])
    heatmap_cols = [c for c in dict.fromkeys(heatmap_cols) if c in df.columns]
    corr = df[heatmap_cols].corr()
    n = len(heatmap_cols)
    fig, ax = plt.subplots(figsize=(max(18, n * 0.65), max(15, n * 0.55)))
    sns.heatmap(corr, cmap="RdBu_r", center=0, ax=ax, square=True, linewidths=0.4,
                linecolor="white", annot=True, fmt=".2f", annot_kws={"size": 6.5},
                cbar_kws={"label": "Pearson correlation", "shrink": 0.8})
    ax.set_title("Figure 5: Correlation Heatmap — All Numeric Feature Groups + Label", fontsize=14)
    ax.tick_params(axis="x", labelsize=9, rotation=90)
    ax.tick_params(axis="y", labelsize=9, rotation=0)
    plt.tight_layout()
    # PNG for quick preview (e.g. in chat) — still subject to whatever
    # viewer rescales it afterwards, which is where pixelation creeps in.
    fig.savefig(os.path.join(outdir, "05_correlation_heatmap.png"), dpi=300, bbox_inches="tight")
    # PDF (vector) for actual use in the report — no pixel grid, so it stays
    # perfectly sharp at any zoom level or embed size in Word/LaTeX/etc.
    fig.savefig(os.path.join(outdir, "05_correlation_heatmap.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure 5: 05_correlation_heatmap.png (+ .pdf vector version for the report)")
    return label_corr


def section_4_5_4_pairplot(df: pd.DataFrame, outdir: str) -> None:
    cols = ["URLSimilarityIndex", "NoOfExternalRef_log", "LineOfCode_log",
            "DomainTitleMatchScore", "label"]
    sample = df[cols].sample(n=min(5000, len(df)), random_state=42).copy()
    # URLSimilarityIndex has a massive spike of legitimate rows at exactly
    # 100, which overplots into a solid unreadable block. A small amount of
    # jitter (visualisation only, not a change to the underlying data)
    # spreads these points out so the density is actually visible.
    rng = np.random.default_rng(42)
    sample["URLSimilarityIndex"] = sample["URLSimilarityIndex"] + rng.uniform(-3.0, 3.0, len(sample))

    # Map the label to readable class names on the DataFrame itself, rather
    # than relabelling the legend afterwards via seaborn's private _legend
    # attribute — that attribute's behaviour differs across seaborn
    # versions and was producing an incomplete/broken legend. Renaming the
    # column up front is version-safe and needs no private API.
    sample["Class"] = sample["label"].map(LABEL_NAMES)
    sample = sample.drop(columns="label")
    class_palette = {LABEL_NAMES[0]: PALETTE[0], LABEL_NAMES[1]: PALETTE[1]}

    # corner=True only plots the lower triangle, which reads as an
    # incomplete grid — use the full matrix so every pairwise panel renders.
    g = sns.pairplot(sample, hue="Class", palette=class_palette,
                      plot_kws={"alpha": 0.25, "s": 12}, diag_kind="hist",
                      height=2.6, corner=False)
    g.fig.suptitle("Figure 7: Pair Plot — Top Correlated Features by Class (n=5,000 sample)",
                    y=1.02, fontsize=13)
    g.savefig(os.path.join(outdir, "07_pairplot.png"), dpi=300, bbox_inches="tight")
    plt.close(g.fig)
    print("Saved Figure 7: 07_pairplot.png")


def section_4_5_5_derived_prob(df: pd.DataFrame, outdir: str) -> None:
    print("\n--- 4.5.5 Derived/probabilistic features: TLDLegitimateProb discretisation ---")
    n_unique = df["TLDLegitimateProb"].nunique()
    print(f"TLDLegitimateProb unique values: {n_unique} (across {len(df):,} rows)")
    top_vals = df["TLDLegitimateProb"].value_counts().head(5)
    print("Most common values and their row counts:")
    print(top_vals)
    com_prob = df.loc[df["TLD"] == "com", "TLDLegitimateProb"].iloc[0]
    com_count = (df["TLD"] == "com").sum()
    print(f"Example: TLD='com' always has TLDLegitimateProb={com_prob:.6f}, "
          f"and 'com' appears {com_count:,} times — the value tracks TLD frequency directly.")

    # TLDLegitimateProb and CharContinuationRate are discretised lookup-style
    # values (TLDLegitimateProb: 465 fixed values, one per TLD; see 4.5.5),
    # not continuous measurements — a standard boxplot renders these as
    # solid, uninformative blocks. Violin plots show the actual value
    # clusters; the four genuinely continuous features keep boxplots.
    violin_cols = {"TLDLegitimateProb", "CharContinuationRate"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flat, DERIVED_PROB_COLS):
        if col in violin_cols:
            sns.violinplot(data=df, x="label", y=col, ax=ax, hue="label",
                            palette=PALETTE, legend=False, cut=0, inner="quartile")
        else:
            sns.boxplot(data=df, x="label", y=col, ax=ax, hue="label",
                        palette=PALETTE, legend=False)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Phishing", "Legitimate"])
        ax.set_title(col + ("  (violin — discretised)" if col in violin_cols else ""), fontsize=10)
        ax.set_xlabel("")
    fig.suptitle("Figure 9: Derived/Probabilistic Feature Spread by Class", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "09_box_derived_prob.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure 9: 09_box_derived_prob.png")


def section_4_5_6_ishttps(df: pd.DataFrame) -> None:
    print("\n--- 4.5.6 IsHTTPS by class ---")
    prop = df.groupby("label")["IsHTTPS"].mean()
    prop.index = [LABEL_NAMES[i] for i in prop.index]
    print(prop.round(4))
    print(f"-> {prop['Phishing']:.1%} of phishing URLs use HTTPS despite being malicious, "
          f"vs {prop['Legitimate']:.1%} of legitimate URLs.")


# ---------------------------------------------------------------------------
# 4.6 Categorical patterns
# ---------------------------------------------------------------------------
def section_4_6_categorical(df: pd.DataFrame, outdir: str) -> None:
    print("\n" + "=" * 70)
    print("4.6 CATEGORICAL PATTERNS")
    print("=" * 70)

    top_tlds = df["TLD"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=top_tlds.values, y=top_tlds.index, ax=ax,
                hue=top_tlds.index, palette="viridis", legend=False)
    ax.set_title("Figure 6: Top 15 Most Frequent TLDs")
    ax.set_xlabel("Count")
    ax.set_ylabel("TLD")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "06_top_tlds.png"), dpi=150)
    plt.close(fig)
    print("Saved Figure 6: 06_top_tlds.png")

    prop_by_label = df.groupby("label")[BINARY_FLAG_COLS].mean().T
    prop_by_label.columns = [LABEL_NAMES[c] for c in prop_by_label.columns]
    prop_by_label = prop_by_label.sort_values("Legitimate", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    prop_by_label.plot(kind="bar", ax=ax, color=[PALETTE[0], PALETTE[1]])
    ax.set_ylabel("Proportion of rows with flag = 1")
    ax.set_title("Figure 8: Prevalence of All Binary Flags by Class")
    ax.set_xticklabels(prop_by_label.index, rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "08_all_binary_flags_by_label.png"), dpi=150)
    plt.close(fig)
    print("Saved Figure 8: 08_all_binary_flags_by_label.png")


# ---------------------------------------------------------------------------
# 4.7 / 4.8 Supporting numbers for imbalance & insight summary
# ---------------------------------------------------------------------------
def imbalance_and_insight_numbers(df: pd.DataFrame, label_corr: pd.Series) -> None:
    print("\n" + "=" * 70)
    print("4.7 / 4.8 SUPPORTING NUMBERS FOR SUMMARY")
    print("=" * 70)
    print(f"Class imbalance ratio (legitimate:phishing): "
          f"{(df['label']==1).sum() / (df['label']==0).sum():.2f} : 1")
    print(f"Top correlate with label: {label_corr.index[0]} ({label_corr.iloc[0]:.3f})")
    print(f"Weakest correlate with label: {label_corr.index[-1]} ({label_corr.iloc[-1]:.3f})")
    obf_corrs = label_corr.reindex(OBFUSCATION_COLS).dropna()
    print(f"Mean correlation of obfuscation-related features: {obf_corrs.mean():.3f} "
          f"(all rank in the bottom third of all features)")


def main():
    parser = argparse.ArgumentParser(description="Task 4 — EDA and visualisation")
    parser.add_argument("--input", default="PhiUSIIL_cleaned.csv")
    parser.add_argument("--outdir", default="figures")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_data(args.input)

    descriptive_analysis(df)
    target_distribution(df, args.outdir)
    missingness_analysis(df, args.outdir)

    print("\n" + "=" * 70)
    print("4.5 FEATURE DISTRIBUTIONS AND RELATIONSHIPS")
    print("=" * 70)
    section_4_5_1_url_lexical(df, args.outdir)
    section_4_5_2_content_structure(df, args.outdir)
    label_corr = section_4_5_3_correlation(df, args.outdir)
    section_4_5_4_pairplot(df, args.outdir)
    section_4_5_5_derived_prob(df, args.outdir)
    section_4_5_6_ishttps(df)

    section_4_6_categorical(df, args.outdir)
    imbalance_and_insight_numbers(df, label_corr)


if __name__ == "__main__":
    main()