"""
Capstone: Does higher pesticide use go with lower (or higher) reported food loss?
Dataset: Global Pesticide Use & Food Loss (FAOSTAT-derived), 133 areas, years 2016/2020/2021.
Modules used: Pandas/NumPy (M3), Matplotlib/Seaborn (M4), SciPy stats (M5).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

DATA_PATHS = [
    "Global Pesticide use & Food Loss Dataset/Dataset.csv",
    "Dataset.csv",
]
ALPHA = 0.05


# ---------------------------------------------------------------- Module 3
def load_data():
    """Load the CSV from the first path that exists and drop exact duplicates."""
    path = next(p for p in DATA_PATHS if os.path.exists(p))
    data = pd.read_csv(path).drop_duplicates()
    print("Rows:", len(data))
    print("Areas:", data["Area"].nunique())
    print("Years:", sorted(data["Year"].unique()))
    print("\nMissing values:")
    print(data.isnull().sum())
    return data


def build_per_capita(data):
    """Total pesticide use per capita, one row per Area-Year."""
    per_capita = data[
        (data["Item"] == "Pesticides (total)")
        & (data["Element"] == "Use per capita")
        & (data["Unit"] == "kg/cap")
    ].copy()
    return per_capita.drop_duplicates(subset=["Area", "Year"])


def build_area_data(per_capita):
    """Average the available years for each area."""
    area_data = per_capita.groupby("Area", as_index=False).agg(
        pesticide_use=("value", "mean"),
        food_loss=("Food loss percentage", "mean"),
        sub_region=("Sub-region Name", "first"),
    )
    # Pesticide use is strongly right-skewed (outliers up to ~4.5 kg/cap),
    # so keep a log version for plotting / robust analysis.
    area_data["log_pesticide"] = np.log10(area_data["pesticide_use"])
    area_data["use_group"] = np.where(
        area_data["pesticide_use"] >= area_data["pesticide_use"].median(),
        "High use",
        "Low use",
    )
    print("\nAreas in comparison:", len(area_data))
    print("Average reported food loss:", round(area_data["food_loss"].mean(), 2), "%")
    return area_data


# ---------------------------------------------------------------- Module 5
def test_normality(area_data):
    """Shapiro-Wilk decides parametric (Pearson) vs non-parametric (Spearman)."""
    print("\n--- Normality (Shapiro-Wilk, H0: data are normal) ---")
    for col in ["pesticide_use", "food_loss"]:
        w, p = stats.shapiro(area_data[col])
        verdict = "normal" if p > ALPHA else "NOT normal"
        print(f"{col}: W={w:.3f}, p={p:.4g} -> {verdict}")


def correlation_test(area_data):
    """Spearman rank correlation with p-value (H0: no monotonic relationship)."""
    rho, p = stats.spearmanr(area_data["pesticide_use"], area_data["food_loss"])
    print("\n--- Spearman correlation ---")
    print(f"rho={rho:.3f}, p={p:.4g}")
    # 0 to +1: higher pesticide use tends to go with higher food loss
    # -1 to 0: higher pesticide use tends to go with lower food loss
    # near 0 : little evidence of a consistent monotonic relationship
    print("Significant at 5%" if p < ALPHA else "Not significant at 5%")
    return rho, p


def group_comparison(area_data):
    """High vs low pesticide users: Levene + Mann-Whitney U on food loss."""
    high = area_data.loc[area_data["use_group"] == "High use", "food_loss"]
    low = area_data.loc[area_data["use_group"] == "Low use", "food_loss"]
    print("\n--- High-use vs Low-use areas (median split) ---")
    print(f"Median food loss: high={high.median():.2f}%, low={low.median():.2f}%")
    lev_stat, lev_p = stats.levene(high, low)
    print(f"Levene (equal variances): stat={lev_stat:.3f}, p={lev_p:.4g}")
    u, p = stats.mannwhitneyu(high, low, alternative="two-sided")
    print(f"Mann-Whitney U: U={u:.1f}, p={p:.4g}")


def region_comparison(area_data):
    """Kruskal-Wallis: does food loss differ between sub-regions?"""
    groups = [
        g["food_loss"].values
        for _, g in area_data.groupby("sub_region")
        if len(g) >= 3
    ]
    h, p = stats.kruskal(*groups)
    print("\n--- Food loss across sub-regions (Kruskal-Wallis) ---")
    print(f"{len(groups)} sub-regions, H={h:.2f}, p={p:.4g}")


def change_over_time(per_capita):
    """Paired Wilcoxon test: 2016 vs 2021 for areas present in both years."""
    wide = per_capita.pivot_table(
        index="Area", columns="Year", values=["value", "Food loss percentage"]
    )
    print("\n--- Change 2016 -> 2021 (paired Wilcoxon) ---")
    for label, col in [("Pesticide use (kg/cap)", "value"),
                       ("Food loss (%)", "Food loss percentage")]:
        pair = wide[col][[2016, 2021]].dropna()
        stat, p = stats.wilcoxon(pair[2016], pair[2021])
        print(f"{label}: n={len(pair)}, median change="
              f"{(pair[2021] - pair[2016]).median():+.3f}, p={p:.4g}")


def time_series_check(per_capita, min_points=8):
    """ARIMA needs a long, regularly spaced series; check before attempting."""
    years = sorted(per_capita["Year"].unique())
    print("\n--- Time-series feasibility (statsmodels ARIMA) ---")
    if len(years) < min_points:
        print(f"Only {len(years)} observation years {years}, non-consecutive.")
        print("Stationarity tests (ADF) and ARIMA forecasting are not valid here, "
              "so they are skipped.")
        return False
    return True


# ---------------------------------------------------------------- Module 4
def plot_distribution(area_data):
    sns.histplot(data=area_data, x="food_loss", bins=15)
    plt.title("Reported Food Loss Across Areas")
    plt.xlabel("Food loss (%)")
    plt.ylabel("Number of areas")
    plt.tight_layout()
    plt.savefig("food_loss_distribution.png", dpi=150)
    plt.show()


def plot_scatter(area_data, rho, p):
    sns.regplot(data=area_data, x="log_pesticide", y="food_loss",
                scatter_kws={"alpha": 0.6})
    plt.title(f"Pesticide Use and Reported Food Loss (Spearman rho={rho:.2f}, p={p:.2f})")
    plt.xlabel("log10 pesticide use per capita (kg/cap)")
    plt.ylabel("Food loss (%)")
    plt.tight_layout()
    plt.savefig("pesticide_and_food_loss.png", dpi=150)
    plt.show()


def plot_regions(area_data):
    region_data = (
        area_data.groupby("sub_region", as_index=False)["food_loss"]
        .mean()
        .sort_values("food_loss")
    )
    sns.barplot(data=region_data, x="food_loss", y="sub_region")
    plt.title("Reported Food Loss by Sub-Region")
    plt.xlabel("Average food loss (%)")
    plt.ylabel("Sub-region")
    plt.tight_layout()
    plt.savefig("food_loss_by_region.png", dpi=150)
    plt.show()


def plot_group_box(area_data):
    sns.boxplot(data=area_data, x="use_group", y="food_loss",
                order=["Low use", "High use"])
    plt.title("Food Loss: Low vs High Pesticide-Use Areas")
    plt.xlabel("Pesticide use per capita (median split)")
    plt.ylabel("Food loss (%)")
    plt.tight_layout()
    plt.savefig("food_loss_by_use_group.png", dpi=150)
    plt.show()



# ------------------------------------------- Pesticide type analysis (M3/M4)
MAIN_TYPES = [
    "Herbicides", "Fungicides and Bactericides", "Insecticides",
    "Other Pesticides nes", "Plant Growth Regulators", "Rodenticides",
]


def pesticide_type_totals(data):
    """Average yearly global use (tonnes) for each pesticide type."""
    use = data[(data["Element"] == "Agricultural Use") & (data["Unit"] == "t")]
    n_years = use["Year"].nunique()
    yearly = use.groupby("Item")["value"].sum() / n_years
    main = yearly[yearly.index.isin(MAIN_TYPES)].sort_values(ascending=False)
    # Sub-types are everything except the total and the main categories
    # (summing both would double count), and generic "nes" buckets.
    sub = yearly.drop(MAIN_TYPES + ["Pesticides (total)"], errors="ignore")
    sub = sub[~sub.index.str.contains("nes")].sort_values(ascending=False).head(10)
    print("\n--- Pesticide types (average global use per year, tonnes) ---")
    for name, val in main.items():
        print(f"{name}: {val:,.0f} t ({val / main.sum() * 100:.1f}% of listed types)")
    return main, sub


def plot_pesticide_types(main, sub):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(x=main.values / 1e3, y=main.index, ax=axes[0], color="#2a7f62")
    axes[0].set_title("Main Pesticide Types")
    axes[0].set_xlabel("Average global use per year (thousand tonnes)")
    axes[0].set_ylabel("")
    for i, v in enumerate(main.values / main.sum() * 100):
        axes[0].text(main.values[i] / 1e3, i, f" {v:.0f}%", va="center")
    sns.barplot(x=sub.values / 1e3, y=sub.index, ax=axes[1], color="#c2703d")
    axes[1].set_title("Top 10 Specific Pesticide Groups")
    axes[1].set_xlabel("Average global use per year (thousand tonnes)")
    axes[1].set_ylabel("")
    fig.suptitle("Which Pesticides Are Used the Most?")
    plt.tight_layout()
    plt.savefig("pesticide_types.png", dpi=150)
    plt.show()

# ---------------------------------------------------------------- main
def main():
    data = load_data()
    per_capita = build_per_capita(data)
    area_data = build_area_data(per_capita)

    test_normality(area_data)
    rho, p = correlation_test(area_data)
    group_comparison(area_data)
    region_comparison(area_data)
    change_over_time(per_capita)
    time_series_check(per_capita)

    plot_distribution(area_data)
    plot_scatter(area_data, rho, p)
    plot_regions(area_data)
    plot_group_box(area_data)
    main_types, sub_types = pesticide_type_totals(data)
    plot_pesticide_types(main_types, sub_types)

    area_data.to_csv("area_level_analysis.csv", index=False)
    print("\nSaved the area-level data and five plot images.")


if __name__ == "__main__":
    main()