from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
import math

# Fetch Dataset
wine_quality = fetch_ucirepo(id=186)

# Setting Display All
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# Convert to DataFrame
df_features = pd.DataFrame(wine_quality.data.features)
df_target = pd.DataFrame(wine_quality.data.targets, columns=["quality"])

# Display Basic Information
print("📌 First 5 Rows of Features:\n", df_features.head())
print("\n🎯 First 5 Rows of Target:\n", df_target.head())
print("\n🔍 Data Info:")
print(df_features.info())
print("\n📊 Descriptive Statistics:\n", df_features.describe())
print("\n⚠ Missing Values Per Column:\n", df_features.isnull().sum())

# Plot All Features Against Target
features = df_features.columns
num_features = len(features)

cols = 3
rows = math.ceil(num_features / cols)
fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
axes = axes.flatten() if num_features > 1 else [axes]

for i, feature in enumerate(features):
    axes[i].scatter(df_features[feature], df_target["quality"], alpha=0.6)
    axes[i].set_title(f"{feature.replace('_', ' ').title()} vs Quality")
    axes[i].set_xlabel(feature.replace('_', ' ').title())
    axes[i].set_ylabel("Quality")
    axes[i].grid(True)

# Delete Empty Subplot
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
