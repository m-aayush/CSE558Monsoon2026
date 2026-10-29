import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'Carlito', 'font.family': 'sans-serif'})

print("--- 1. Loading Dataset ---")
# Downloaded from kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants
df = pd.read_csv("zomato.csv")
print(f"Raw shape: {df.shape}")

print("\n--- 2. Data Preparation ---")
# Clean 'rate' column
def clean_rate(val):
    if pd.isna(val) or val in ['NEW', '-']:
        return np.nan
    try:
        return float(str(val).split('/')[0].strip())
    except:
        return np.nan

df['rate_clean'] = df['rate'].apply(clean_rate)
new_count = df['rate'].isin(['NEW']).sum()
unrated_count = df['rate'].isin(['-', np.nan]).sum()

# Drop rows with 'NEW' ratings
df_clean = df[~df['rate'].isin(['NEW'])].copy()

# Handle missing/dash ratings via imputation and flagging
df_clean['rate_imputed_flag'] = df_clean['rate_clean'].isna()
location_medians = df_clean.groupby('location')['rate_clean'].transform('median')
df_clean['rate_clean'] = df_clean['rate_clean'].fillna(location_medians)

# Drop duplicates on (name, address)
df_clean = df_clean.drop_duplicates(subset=['name', 'address'])

# Clean 'cost_two' ("approx_cost(for twopeople)")
cost_col = 'approx_cost(for two people)' if 'approx_cost(for two people)' in df_clean.columns else df_clean.columns[10]
df_clean['cost_two'] = df_clean[cost_col].astype(str).str.replace(',', '').str.replace('nan', '0')
df_clean['cost_two'] = pd.to_numeric(df_clean['cost_two'], errors='coerce').fillna(0)

# Cap outliers in cost_two using IQR
Q1 = df_clean['cost_two'].quantile(0.25)
Q3 = df_clean['cost_two'].quantile(0.75)
IQR = Q3 - Q1
upper_fence = Q3 + 1.5 * IQR
outliers_cost = (df_clean['cost_two'] > upper_fence).sum()
df_clean['cost_two'] = np.where(df_clean['cost_two'] > upper_fence, upper_fence, df_clean['cost_two'])

# One-hot encoding for location and cuisines
df_encoded = pd.get_dummies(df_clean, columns=['location', 'cuisines'], drop_first=True)
rows_enc, cols_enc = df_encoded.shape
total_cells = rows_enc * cols_enc

print(f"After preparation shape: {rows_enc} rows, {cols_enc} attributes")
print(f"Instances x Attributes = {total_cells:,} (Requirement >= 3,000,000: {total_cells >= 3000000})")

print("\n--- 3. Generating EDA Plots ---")
# Plot 1: Ratings distribution
plt.figure(figsize=(6, 3.5))
sns.histplot(df_clean['rate_clean'].dropna(), bins=20, color='#7396C4', kde=True)
plt.title("Distribution of Restaurant Ratings")
plt.xlabel("Mean Customer Rating (out of 5)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("fig_ratings.pdf")
plt.close()

# Plot 2: Outlet density per neighbourhood
loc_counts = df_clean['location'].value_counts()
plt.figure(figsize=(6, 3.5))
sns.histplot(loc_counts, bins=25, color='#4A6FA5', kde=True)
plt.title("Outlets per Neighbourhood Distribution")
plt.xlabel("Number of Outlets")
plt.ylabel("Neighbourhood Count")
plt.tight_layout()
plt.savefig("fig_density.pdf")
plt.close()
print("EDA plots saved successfully.")

print("\n--- 4. Statistical Inferences (Hypothesis Tests) ---")
# Filter valid observed ratings for testing (excluding imputed ones)
df_valid = df_clean[~df_clean['rate_imputed_flag']].copy()

# Test 1: P1 - Crowded vs Least Crowded Neighbourhoods (North Indian, INR 400-800)
subset_p1 = df_valid[
    df_valid['cuisines'].str.contains('North Indian', na=False) &
    (df_valid['cost_two'] >= 400) & (df_valid['cost_two'] <= 800)
].copy()

loc_outlet_counts = df_valid['location'].value_counts()
subset_p1['outlet_density'] = subset_p1['location'].map(loc_outlet_counts)
q_low = subset_p1['outlet_density'].quantile(0.33)
q_high = subset_p1['outlet_density'].quantile(0.67)

group_D = subset_p1[subset_p1['outlet_density'] >= q_high]['rate_clean']
group_S = subset_p1[subset_p1['outlet_density'] <= q_low]['rate_clean']

t_stat_1, p_val_1 = stats.ttest_ind(group_D, group_S, equal_var=False, alternative='less')
print(f"Test 1 (P1) Welch t-test: t = {t_stat_1:.2f}, p-value = {p_val_1:.2e}")

# Test 2: P2 - Online Ordering in Crowded Neighbourhoods
subset_p2 = subset_p1[subset_p1['outlet_density'] >= q_high]
group_on = subset_p2[subset_p2['online_order'] == 'Yes']['rate_clean']
group_off = subset_p2[subset_p2['online_order'] == 'No']['rate_clean']

t_stat_2, p_val_2 = stats.ttest_ind(group_on, group_off, equal_var=False, alternative='greater')
print(f"Test 2 (P2) Welch t-test: t = {t_stat_2:.2f}, p-value = {p_val_2:.4f}")

# Test 3: P3 - Table Booking impact across all restaurants
group_book = df_valid[df_valid['book_table'] == 'Yes']['rate_clean']
group_nobook = df_valid[df_valid['book_table'] == 'No']['rate_clean']

t_stat_3, p_val_3 = stats.ttest_ind(group_book, group_nobook, equal_var=False, alternative='greater')
print(f"Test 3 (P3) Welch t-test: t = {t_stat_3:.2f}, p-value = {p_val_3:.2e}")
print("Analysis complete.")
