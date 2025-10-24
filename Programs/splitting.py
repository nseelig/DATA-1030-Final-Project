import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

PATH = r"C:\Users\nseel\CS\DATA1030\Final Project\Cleaned Data\ca_all_years.csv"
df = pd.read_csv(PATH)

df = df[df["StudentGroup"].str.strip().eq("All Students")].copy()
for c in ["ProficientOrAbove_percent", "StudentSubGroup_TotalTested", "ParticipationRate", "SchYear"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
if "SchYear" in df.columns:
    df = df[df["SchYear"] != 2020]
y = df["ProficientOrAbove_percent"]
keep = y.notna()
if "StudentSubGroup_TotalTested" in df.columns:
    keep &= df["StudentSubGroup_TotalTested"] > 0
df, y = df[keep].copy(), y[keep]
num_feats = [c for c in ["ParticipationRate", "StudentSubGroup_TotalTested", "SchYear"] if c in df.columns]
X = df[num_feats].copy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=df["SchYear"]
)
pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler())
])
X_train_pre = pipe.fit_transform(X_train)
X_test_pre  = pipe.transform(X_test)
print("Shapes (preprocessed):", X_train_pre.shape, X_test_pre.shape)


