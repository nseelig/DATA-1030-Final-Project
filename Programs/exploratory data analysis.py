import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Load and setup ---
PATH = r"C:\Users\nseel\CS\DATA1030\Final Project\Cleaned Data\ca_all_years.csv"
df = pd.read_csv(PATH)

save_dir = r"C:\Users\nseel\CS\DATA1030\Final Project\Visualizations"
os.makedirs(save_dir, exist_ok=True)

# Coerce numerics
for col in ["ProficientOrAbove_count", "StudentSubGroup_TotalTested",
            "ProficientOrAbove_percent", "ParticipationRate"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Helper: insert NaNs for missing years (like 2020)
def with_year_gaps(years, values):
    x, y = [], []
    for i in range(len(years)):
        x.append(years[i]); y.append(values[i])
        if i < len(years)-1 and (years[i+1] - years[i] > 1):
            x.append(years[i]+1); y.append(np.nan)
    return x, y

# ================================================================
# Aggregate Proficiency Over Time (Main Line Graph)
# ================================================================
all_df = df[df["StudentSubGroup"] == "All Students"]
agg_all = (all_df.groupby("SchYear")
            .agg(Proficient=("ProficientOrAbove_count", "sum"),
                 Tested=("StudentSubGroup_TotalTested", "sum"))
            .reset_index())
agg_all["Rate"] = 100 * agg_all["Proficient"] / agg_all["Tested"]

years, rates = agg_all["SchYear"].to_numpy(), agg_all["Rate"].to_numpy()
x, y = with_year_gaps(years, rates)

plt.figure(figsize=(8,5))
plt.plot(x, y, marker="o", linewidth=2)
plt.title("California Grade 4 Math: % Proficient or Above (2015–2024)")
plt.xlabel("School Year"); plt.ylabel("Proficiency Rate (%)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "aggregate_trend.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: aggregate_trend.png")

# ================================================================
# Boxplot of District Proficiency by Year
# ================================================================
plt.figure(figsize=(10,6))
box = df[(df["StudentSubGroup"]=="All Students") & df["ProficientOrAbove_percent"].notna()]
years = sorted(box["SchYear"].unique())
data = [ (box.loc[box["SchYear"]==y, "ProficientOrAbove_percent"]*100).values for y in years ]
plt.boxplot(data, labels=years, patch_artist=True)
plt.title("Distribution of District Proficiency (All Students, 2015–2024)")
plt.xlabel("School Year"); plt.ylabel("Proficiency Rate (%)")
plt.grid(True, axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "district_boxplot.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: district_boxplot.png")

# ================================================================
# Change from 2015 → 2024 (by Subgroup)
# ================================================================
base = df[df["SchYear"].isin([2015,2024])]
agg_change = (base.groupby(["SchYear","StudentSubGroup"])
                   .agg(Rate=("ProficientOrAbove_percent","mean"))
                   .reset_index())
pivot = agg_change.pivot(index="StudentSubGroup", columns="SchYear", values="Rate").dropna()
pivot["Change"] = (pivot[2024] - pivot[2015]) * 100
pivot = pivot.sort_values("Change")

plt.figure(figsize=(9,6))
plt.barh(pivot.index, pivot["Change"], color=np.where(pivot["Change"]>0,"green","red"))
plt.title("Change in Average Proficiency, 2015 → 2024 (by Subgroup)")
plt.xlabel("Percentage Point Change")
plt.axvline(0, color="black", lw=1)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "subgroup_change_2015_2024.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: subgroup_change_2015_2024.png")

# ================================================================
# Participation vs. Proficiency
# ================================================================
temp = df[df["StudentSubGroup"]=="All Students"].dropna(subset=["ParticipationRate","ProficientOrAbove_percent"])
plt.figure(figsize=(8,6))
plt.scatter(temp["ParticipationRate"], temp["ProficientOrAbove_percent"]*100, alpha=0.4)
plt.title("Participation vs. Proficiency (All Students, 2015–2024)")
plt.xlabel("Participation Rate (%)"); plt.ylabel("Proficiency Rate (%)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "participation_vs_proficiency.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: participation_vs_proficiency.png")

# ================================================================
# Heatmap (Subgroup × Year)
# ================================================================
subset = ["Asian","Black or African American","Hispanic or Latino","White",
           "Economically Disadvantaged","Not Economically Disadvantaged"]
pivot = (df[df["StudentSubGroup"].isin(subset)]
         .groupby(["StudentSubGroup","SchYear"])["ProficientOrAbove_percent"]
         .mean().unstack())

plt.figure(figsize=(10,5))
plt.imshow(pivot*100, cmap="YlGnBu", aspect="auto")
plt.xticks(range(len(pivot.columns)), pivot.columns)
plt.yticks(range(len(pivot.index)), pivot.index)
plt.colorbar(label="Proficiency Rate (%)")
plt.title("Proficiency by Subgroup and Year (Heatmap)")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "heatmap_subgroup.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: heatmap_subgroup.png")

# ================================================================
# Histogram of District Proficiency (2024)
# ================================================================
subset = df[(df["SchYear"]==2024) & (df["StudentSubGroup"]=="All Students")]
plt.figure(figsize=(8,5))
plt.hist(subset["ProficientOrAbove_percent"]*100, bins=20, edgecolor="black")
plt.title("Distribution of District Proficiency (All Students, 2024)")
plt.xlabel("Proficiency Rate (%)"); plt.ylabel("Number of Districts")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "histogram_districts_2024.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: histogram_districts_2024.png")

# ================================================================
# 7️⃣  Year-over-Year Change Bar Chart
# ================================================================
agg = (df[df["StudentSubGroup"]=="All Students"]
       .groupby("SchYear")["ProficientOrAbove_percent"].mean().reset_index())
agg["YoY_Change"] = agg["ProficientOrAbove_percent"].diff()*100

plt.figure(figsize=(9,5))
plt.bar(agg["SchYear"], agg["YoY_Change"], color=np.where(agg["YoY_Change"]>0,"green","red"))
plt.axhline(0, color="black", lw=1)
plt.title("Year-over-Year Change in Proficiency (All Students)")
plt.xlabel("School Year"); plt.ylabel("Δ Proficiency (p.p.)")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "yoy_change.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: yoy_change.png")

# ================================================================
# District Size vs. Proficiency (2024)
# ================================================================
latest = df[(df["SchYear"]==2024) & (df["StudentSubGroup"]=="All Students")]
plt.figure(figsize=(8,6))
plt.scatter(latest["StudentSubGroup_TotalTested"], latest["ProficientOrAbove_percent"]*100, alpha=0.5)
plt.xscale("log")
plt.title("District Size vs. Proficiency (All Students, 2024)")
plt.xlabel("Total Tested (log scale)"); plt.ylabel("Proficiency Rate (%)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "district_size_vs_proficiency.png"), dpi=300, bbox_inches="tight")
plt.show()
print("Saved: district_size_vs_proficiency.png")
