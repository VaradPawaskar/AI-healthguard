#%%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

RANDOM_STATE = 42
TEST_SIZE = 0.2

DATA_PATH = "processed.cleveland.data"
TARGET_COL = "target"

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", TARGET_COL
]

df = pd.read_csv(
    DATA_PATH,
    header=None,
    names=columns,
    na_values=["?"]
)

print("Raw shape:", df.shape)
display(df.head())

print("\nMissing values before cleaning:")
print(df.isna().sum())

# Original target has values 0,1,2,3,4 -> make it binary (0=no disease, 1=disease)
if df[TARGET_COL].nunique() > 2:
    df[TARGET_COL] = (df[TARGET_COL] > 0).astype(int)
    print("\nConverted multi-class target to binary (0 = no disease, 1 = disease).")

# Drop duplicates
before = df.shape[0]
df = df.drop_duplicates()
after = df.shape[0]
print(f"\nDropped {before - after} duplicate rows.")

# Handle missing values
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in df.columns if c not in numeric_cols]

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values after cleaning:")
print(df.isna().sum())

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

print("\nFeature columns:")
print(list(X.columns))

print("\nTarget distribution:")
print(y.value_counts())


# 2. EDA – CLASS BALANCE & CORRELATION

# Class distribution
class_counts = y.value_counts().sort_index()
plt.figure()
sns.barplot(x=class_counts.index.astype(str), y=class_counts.values)
plt.title("Target Class Distribution (0 = no disease, 1 = disease)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# Correlation heatmap
numeric_df = df.select_dtypes(include=[np.number])
corr = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", annot=True, fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

target_corr = corr[TARGET_COL].sort_values(ascending=False)
print("\nTop correlations with target:")
print(target_corr)

# Pairplot of a few features
sample_features = X.columns[:4]
sns.pairplot(df[list(sample_features) + [TARGET_COL]], hue=TARGET_COL)
plt.suptitle("Pairplot of Sample Features", y=1.02)
plt.show()

#TEmporary
print("Unique CP values:", X['cp'].unique())
print("Average target by CP:", df.groupby('cp')['target'].mean())

# %%
