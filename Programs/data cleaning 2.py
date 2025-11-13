import pandas as pd
df = pd.read_csv(r"C:\Users\nseel\CS\DATA1030\DATA-1030-Final-Project\Cleaned Data\ca_all_years.csv")
df.columns = df.columns.str.strip()

exclude_groups = [
    "Disability Status", "EL Status", "Migrant Status",
    "Homeless Enrolled Status", "Military Connected Status",
    "Foster Care Status"
]
df = df[~df["StudentGroup"].isin(exclude_groups)]
num_cols = [
    "ProficientOrAbove_percent",
    "ProficientOrAbove_count",
    "StudentGroup_TotalTested",
    "StudentSubGroup_TotalTested",
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
pct_wide = df.pivot_table(
    index=["SchYear", "DistName"],
    columns=["StudentGroup", "StudentSubGroup"],
    values="ProficientOrAbove_percent",
    aggfunc="mean"
)
count_wide = df.pivot_table(
    index=["SchYear", "DistName"],
    columns=["StudentGroup", "StudentSubGroup"],
    values="ProficientOrAbove_count",
    aggfunc="sum"  
)
group_total_wide = df.pivot_table(
    index=["SchYear", "DistName"],
    columns=["StudentGroup", "StudentSubGroup"],
    values="StudentGroup_TotalTested",
    aggfunc="sum"
)
subgroup_total_wide = df.pivot_table(
    index=["SchYear", "DistName"],
    columns=["StudentGroup", "StudentSubGroup"],
    values="StudentSubGroup_TotalTested",
    aggfunc="sum"
)
pct_wide.columns = [f"{g}_{sg}_ProficientPct" for g, sg in pct_wide.columns]
count_wide.columns = [f"{g}_{sg}_ProficientCount" for g, sg in count_wide.columns]
group_total_wide.columns = [f"{g}_{sg}_GroupTotalTested" for g, sg in group_total_wide.columns]
subgroup_total_wide.columns = [f"{g}_{sg}_SubgroupTotalTested" for g, sg in subgroup_total_wide.columns]
wide_df = (
    pd.concat([pct_wide, count_wide, group_total_wide, subgroup_total_wide], axis=1)
      .reset_index()
)
rename_dict = {
    'SchYear': 'Year',
    'DistName': 'District',
    'All Students_All Students_ProficientPct': 'All_Pct',
    'Economic Status_Economically Disadvantaged_ProficientPct': 'EconDisadv_Pct',
    'Economic Status_Not Economically Disadvantaged_ProficientPct': 'NotEconDisadv_Pct',
    'Gender_Female_ProficientPct': 'Female_Pct',
    'Gender_Male_ProficientPct': 'Male_Pct',
    'RaceEth_American Indian or Alaska Native_ProficientPct': 'AIAN_Pct',
    'RaceEth_Asian_ProficientPct': 'Asian_ProficientPct',
    'RaceEth_Black or African American_ProficientPct': 'Black_Pct',
    'RaceEth_Filipino_ProficientPct': 'Filipino_ProficientPct',
    'RaceEth_Hispanic or Latino_ProficientPct': 'Hisp_Pct',
    'RaceEth_Native Hawaiian or Pacific Islander_ProficientPct': 'NHPI_Pct',
    'RaceEth_Two or More_ProficientPct': 'TwoOrMore_Pct',
    'RaceEth_White_ProficientPct': 'White_Pct',

    'All Students_All Students_ProficientCount': 'All_Count',
    'Economic Status_Economically Disadvantaged_ProficientCount': 'EconDisadv_Count',
    'Economic Status_Not Economically Disadvantaged_ProficientCount': 'NotEconDisadv_Count',
    'Gender_Female_ProficientCount': 'Female_Count',
    'Gender_Male_ProficientCount': 'Male_Count',
    'RaceEth_American Indian or Alaska Native_ProficientCount': 'AIAN_Count',
    'RaceEth_Asian_ProficientCount': 'Asian_Count',
    'RaceEth_Black or African American_ProficientCount': 'Black_Count',
    'RaceEth_Filipino_ProficientCount': 'Filipino_ProficientCount',
    'RaceEth_Hispanic or Latino_ProficientCount': 'Hisp_Count',
    'RaceEth_Native Hawaiian or Pacific Islander_ProficientCount': 'NHPI_Count',
    'RaceEth_Two or More_ProficientCount': 'TwoOrMore_Count',
    'RaceEth_White_ProficientCount': 'White_Count',

    'All Students_All Students_GroupTotalTested': 'All_GroupTested',
    'Economic Status_Economically Disadvantaged_GroupTotalTested': 'EconDisadv_GroupTested',
    'Economic Status_Not Economically Disadvantaged_GroupTotalTested': 'NotEconDisadv_GroupTested',
    'Gender_Female_GroupTotalTested': 'Female_GroupTested',
    'Gender_Male_GroupTotalTested': 'Male_GroupTested',
    'RaceEth_American Indian or Alaska Native_GroupTotalTested': 'AIAN_GroupTested',
    'RaceEth_Asian_GroupTotalTested': 'Asian_GroupTested',
    'RaceEth_Black or African American_GroupTotalTested': 'Black_GroupTested',
    'RaceEth_Filipino_GroupTotalTested': 'Filipino_GroupTested',
    'RaceEth_Hispanic or Latino_GroupTotalTested': 'Hisp_GroupTested',
    'RaceEth_Native Hawaiian or Pacific Islander_GroupTotalTested': 'NHPI_GroupTested',
    'RaceEth_Two or More_GroupTotalTested': 'TwoOrMore_GroupTested',
    'RaceEth_White_GroupTotalTested': 'White_GroupTested',

    'All Students_All Students_SubgroupTotalTested': 'All_SubgroupTested',
    'Economic Status_Economically Disadvantaged_SubgroupTotalTested': 'EconDisadv_SubgroupTested',
    'Economic Status_Not Economically Disadvantaged_SubgroupTotalTested': 'NotEconDisadv_SubgroupTested',
    'Gender_Female_SubgroupTotalTested': 'Female_SubgroupTested',
    'Gender_Male_SubgroupTotalTested': 'Male_SubgroupTested',
    'RaceEth_American Indian or Alaska Native_SubgroupTotalTested': 'AIAN_SubgroupTested',
    'RaceEth_Asian_SubgroupTotalTested': 'Asian_SubgroupTested',
    'RaceEth_Black or African American_SubgroupTotalTested': 'Black_SubgroupTested',
    'RaceEth_Filipino_SubgroupTotalTested': 'Filipino_SubgroupTested',
    'RaceEth_Hispanic or Latino_SubgroupTotalTested': 'Hisp_SubgroupTested',
    'RaceEth_Native Hawaiian or Pacific Islander_SubgroupTotalTested': 'NHPI_SubgroupTested',
    'RaceEth_Two or More_SubgroupTotalTested': 'TwoOrMore_SubgroupTested',
    'RaceEth_White_SubgroupTotalTested': 'White_SubgroupTested'
}
wide_df.rename(columns=rename_dict, inplace=True)
wide_df['District'] = wide_df['District'].str.replace('School District', '', regex=False).str.strip()
wide_df.to_csv(
    r"C:\Users\nseel\CS\DATA1030\DATA-1030-Final-Project\Cleaned Data\ca_all_years_wide.csv",
    index=False
)
print(wide_df.head())
# Count how many times each district appears
district_counts = wide_df['District'].value_counts()

# Filter districts with fewer than 9 rows
rare_districts = district_counts[district_counts < 2].index

# Print them
print("Districts with fewer than 9 rows:")
count = 0
for district in rare_districts:
    print(district)
    count += 1
print(count)



