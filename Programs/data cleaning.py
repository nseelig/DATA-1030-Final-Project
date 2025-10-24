import pandas as pd 
import os 
import glob 

years = ["15", "16", "17", "18", "19", "21", "22", "23", "24"] 
input_dir = r"C:\Users\nseel\CS\DATA1030\Final Project\Raw Data" 
output_dir = r"C:\Users\nseel\CS\DATA1030\Final Project\Cleaned Data" 
os.makedirs(output_dir, exist_ok=True) 
cols_to_keep = [ "SchYear", "DistName", "StudentGroup", "StudentGroup_TotalTested", "StudentSubGroup", "StudentSubGroup_TotalTested", "ProficientOrAbove_count", "ProficientOrAbove_percent", "ParticipationRate", "DistLocale" ] 
for i in years: 
    input_path = os.path.join(input_dir, f"edc-2.1-california-20{i}.csv") 
    df = pd.read_csv(input_path) 
    df = df[df["DataLevel"] == "District"] 
    df = df[df["Subject"] == "math"] 
    df = df[df["GradeLevel"] == "G04"] 
    year = "20" + i 
    year = int(year) 
    df["SchYear"] = year 
    df = df[cols_to_keep] 
    df["ProficientOrAbove_count"] = pd.to_numeric(df["ProficientOrAbove_count"], errors="coerce")
    df["ProficientOrAbove_percent"] = pd.to_numeric(df["ProficientOrAbove_percent"], errors="coerce")
    df["ParticipationRate"] = pd.to_numeric(df["ParticipationRate"], errors="coerce")
    output_path = os.path.join(output_dir, f"ca20{i}.csv") 
    df.to_csv(output_path, index=False) 
all_files = glob.glob(os.path.join(output_dir, "ca20*.csv")) 
ca_all = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True) 
ca_all.to_csv(os.path.join(output_dir, "ca_all_years.csv"), index=False)