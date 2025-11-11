"""
IVF Success Prediction - Phase 1: Real Kaggle Dataset Exploration
Working with actual IVF data from Kaggle!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100 

# =============================================================================
# STEP 1: LOAD THE REAL KAGGLE DATASET
# =============================================================================
print("=" * 70)
print("LOADING REAL KAGGLE IVF DATASET")
print("=" * 70)

# Load the dataset 
df = pd.read_csv("kaggle_data_set.csv")

print(f"\n✓ Dataset loaded successfully!")
print(f"   Total patients: {len(df)}")
print(f"   Total features: {len(df.columns)}")

# =============================================================================
# STEP 2: INITIAL DATA EXPLORATION
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: INITIAL DATA EXPLORATION")
print("-" * 70)

# 2.1 First look at the data
print("\n1. FIRST 5 ROWS:")
print(df.head())

# 2.2 Column names and types 
print("\n2. COLUMN INFORMATION:")
print(df.info())

# 2.3: Shape
print(f"\n3. DATASET SHAPE:")
print(f"    Rows (patients): {df.shape[0]}")
print(f"    Columns (features + outcome): {df.shape[1]}")

# 2.4: Summary statistics
print("\n4. SUMMARY STATISTICS (Key Numerical Features):")
key_numeric_cols = ['Age', 'BMI', 'AMH_Level', 'FSH_Level', 'Endometrial_Thickness_mm', 'AFC_Count', 'Number_of_Embryos_Transferred']
print(df[key_numeric_cols].describe().round(2))

# =============================================================================
# STEP 3: CHECK DATA QUALITY
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: DATA QUALITY CHECK")
print("-" * 70)

# 3.1 Missing values
print("\n1. MISSING VALUES:")
missing = df.isnull().sum()
missing_pct = (missing/len(df)*100).round(2)
missing_df = pd.DataFrame({
    "Missing_Count": missing, 
    "Percentage" : missing_pct
})
missing_df = missing_df[missing_df["Missing_Count"]>0].sort_values("Missing_Count", ascending=False)

if len(missing_df) > 0: 
    print(missing_df)
    print(f"\n ⚠ Found {len(missing_df)} columns with missing data")
else: 
    print("   ✓ No missing values detected!")

# 3.2 Check for duplicates
duplicates = df.duplicated().sum()
print(f"\n2. DUPLICATE ROWS: {duplicates}")
if duplicates > 0: 
    print(f" ⚠ Warning: {duplicates} duplicate rows found")

# 3.3 Check outcome variable (TARGET) 
print("\n3 TARGET VARIABLE: 'Live_Birth_Success'")
if 'Live_Birth_Success' in df.columns: 
    outcome_counts = df['Live_Birth_Success'].value_counts()
    print(f"\n    Distribution:")
    print(f"    Failed (0): {outcome_counts.get(0,0)} ({outcome_counts.get(0,0)/len(df)*100:.1f}%)")
    print(f"    Success (1): {outcome_counts.get(1,0)} ({outcome_counts.get(1,0)/len(df)*100:.1f}%)")

    # Check for class imbalance
    imbalance_ratio = outcome_counts.max() / outcome_counts.min()
    if imbalance_ratio > 2: 
        print(f"   ⚠ CLASS IMBALANCE detected (ratio: {imbalance_ratio:.2f}:1)")
        print("   → Will need to use class_weight='balanced' in models")
    else:
        print(f"   ✓ Classes reasonably balanced (ratio: {imbalance_ratio:.2f}:1)")
else:
    print("   ✗ ERROR: 'Live_Birth_Success' column not found!")

# =============================================================================
# STEP 4: UNDERSTAND THE FEATURES
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: FEATURE UNDERSTANDING")
print("-" * 70)

# Identify feature types
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# Remove ID and outcome from feature lists
if 'Patient_ID' in numeric_features:
    numeric_features.remove('Patient_ID')
if 'Live_Birth_Success' in numeric_features:
    numeric_features.remove('Live_Birth_Success')
if 'Implantation_Success' in numeric_features:
    numeric_features.remove('Implantation_Success')

print(f"\n1. NUMERICAL FEATURES ({len(numeric_features)}):")
for i, feat in enumerate(numeric_features[:10],1):
    print(f"    {i:2}. {feat}")
if len(numeric_features) > 10: 
    print(f"    ... and {len(numeric_features) - 10} more")

print(f"\n2. CATEGORICAL FEATURES ({len(categorical_features)}):")
for i, feat in enumerate(categorical_features, 1):
    unique_vals = df[feat].nunique()
    print(f"   {i:2}. {feat:30} ({unique_vals} unique values)")

# Show example categorical distributions
print("n\3. KEY CATEGORICAL DISTRIBUTIONS: ")

# Diagnosis Type
if 'Diagnosis_Type' in df.columns: 
    print("\n   Diagnosis Type:")
    print(df['Diagnosis_Type'].value_counts().head())

# Transfer TYpe
if 'Transfer_Type' in df.columns: 
    print("\n Transfer Type: ")
    print(df['Transfer_Type'].value_counts())

# ICSI vs. IVF 
if 'ICSI_or_IVF' in df.columns: 
    print("\n   ICSI vs IVF:")
    print(df['ICSI_or_IVF'].value_counts())

# =============================================================================
# STEP 5: OUTLIER DETECTION
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: OUTLIER DETECTION")
print("-" * 70)

"""
Check for unrealistic values that might indicate data quality issues
"""
print("\n Checking for potential outliers in key features:")

# Age - should be 18-50 for IVF
if 'Age' in df.columns:
    age_issues = df[(df['Age'] < 18 ) | (df['Age'] > 50)]
    print(df['Age'].min())
    print(df['Age'].max())
    print(f"\n   Age outside 18-50 range: {len(age_issues)} patients")
    if len(age_issues) > 0: 
        print(f"   Range: {df['Age'].min():.1f} - {df['Age'].max():.1f}")

# BMI - should be 15-50
if 'BMI' in df.columns:
    bmi_issues = df[(df['BMI'] < 15) | (df['BMI'] > 50)]
    print(df['BMI'].min())
    print(df['BMI'].max())
    print(f"\n    BMI outside 15-50 range: {len(bmi_issues)} patients")
    if len(bmi_issues) > 0: 
        print(f"   Range: {df['BMI'].min():.1f} - {df['BMI'].max():.1f}")

# AMH - should be 0-15 ng/mL
if 'AMH_Level' in df.columns: 
    amh_issues = df[(df['AMH_Level'] < 0) | (df['AMH_Level'] > 15)]
    print(df['AMH_Level'].min())
    print(df['AMH_Level'].max())
    print(f"   AMH outside 0-15 range: {len(amh_issues)} patients")
    if len(amh_issues) > 0: 
        print(f"   Range: {df['AMH_Level'].min():.1f} - {df['AMH_Level'].max():.1f}")

# =============================================================================
# STEP 6: CORRELATION ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: CORRELATION ANALYSIS")
print("-" * 70)

"""
Identify which features are most correlated with success
"""
if 'Live_Birth_Success' in df.columns: 
    # Select numeric columns for correlation
    numeric_cols = df[numeric_features + ['Live_Birth_Success']].columns
    correlation_with_outcome = df[numeric_cols].corr()['Live_Birth_Success'].sort_values(ascending=False)

    print("\n TOP 10 FEATURES POSITIVELY CORRELATED WITH SUCCESS:")
    print(correlation_with_outcome.head(11)[1:11])

    print("n\TOP 10 FEATURES NEGATIVELY CORRELATED WITH SUCCESS:")
    print(correlation_with_outcome.tail(10))

# =============================================================================
# STEP 7: VISUALIZATIONS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: CREATING VISUALIZATIONS")
print("-" * 70)

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 7.1: Age Distribution 
ax1 = fig.add_subplot(gs[0,0])
ax1.hist(df['Age'], bins=30, edgecolor="black", alpha=0.7, color='steelblue')
ax1.set_title('Age Distribution', fontsize=12, fontweight='bold')
ax1.set_xlabel('Age (years)')
ax1.set_ylabel('Count')
ax1.axvline(df['Age'].mean(), color="red", linestyle='--', label=f"Mean: {df['Age'].mean():.1f}")
ax1.legend()

# 7.2: Success Rate by Age Group
ax2 = fig.add_subplot(gs[0,1])
age_bins = pd.cut(df['Age'], bins=[20,25,30,35,40,45], labels=["20-25", "25-30", "30-35", "35-40", "40-45"])
success_by_age = df.groupby(age_bins)["Live_Birth_Success"].agg(['mean', 'count'])
ax2.bar(range(len(success_by_age)), success_by_age['mean'], color='coral', alpha=0.7)
ax2.set_xticks(range(len(success_by_age)))
ax2.set_xticklabels(success_by_age.index, rotation=45)
ax2.set_title('Success Rate by Age Group', fontsize=12, fontweight='bold')
ax2.set_ylabel('Success Rate')
ax2.set_ylim([0,1])
# Add count labels
for i, (idx, row) in enumerate(success_by_age.iterrows()): 
    ax2.text(i, row['mean'] + 0.02, f"n={row['count']}", ha='center', fontsize=8)

# 7.3: AMH Distribution
ax3 = fig.add_subplot(gs[0,2])
ax3.hist(df["AMH_Level"], bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
ax3.set_title('AMH Level Distribution', fontsize=12, fontweight='bold')
ax3.set_xlabel('AMH (ng/mL)')
ax3.set_ylabel('Count')
ax3.axvline(df['AMH_Level'].mean(), color='red', linestyle='--', label=f"Mean: {df["AMH_Level"].mean():.2f}")
ax3.legend()

# 7.4: Success Rate by Diagnosis
ax4 = fig.add_subplot(gs[1,0])
if 'Diagnosis_Type' in df.columns: 
    diagnosis_success = df.groupby("Diagnosis_Type").agg({
        'Live_Birth_Success': ["mean", "count"]
    }).sort_values(('Live_Birth_Success', 'mean'), ascending=True)

    ax4.barh(range(len(diagnosis_success)), diagnosis_success[('Live_Birth_Success', 'mean')], 
             color='purple', alpha=0.7)
    ax4.set_yticks(range(len(diagnosis_success)))
    ax4.set_yticklabels(diagnosis_success.index)
    ax4.set_title('Success Rate by Diagnosis', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Success Rate')
    ax4.set_xlim([0,1])

# 7.5: BMI Distribution
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(df['BMI'], bins=30, edgecolor='black', alpha=0.7, color='orange')
ax5.set_title('BMI Distribution', fontsize=12, fontweight='bold')
ax5.set_xlabel('BMI')
ax5.set_ylabel('Count')
ax5.axvline(df['BMI'].mean(), color='red', linestyle='--', label=f"Mean: {df['BMI'].mean():.1f}")
ax5.legend()

# 7.6: Embryo Grade Distribution
ax6 = fig.add_subplot(gs[1, 2])
if 'Embryo_Grade' in df.columns:
    grade_counts = df['Embryo_Grade'].value_counts().sort_index()
    ax6.bar(range(len(grade_counts)), grade_counts.values, color='teal', alpha=0.7)
    ax6.set_xticks(range(len(grade_counts)))
    ax6.set_xticklabels(grade_counts.index)
    ax6.set_title('Embryo Grade Distribution', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Embryo Grade')
    ax6.set_ylabel('Count')

# 7.7: Correlation Heatmap (Top Features)
ax7 = fig.add_subplot(gs[2, :2])
# Select top correlated features
if 'Live_Birth_Success' in df.columns:
    top_features = correlation_with_outcome.abs().sort_values(ascending=False).head(11).index[1:]  # Exclude outcome itself
    corr_matrix = df[list(top_features) + ['Live_Birth_Success']].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                ax=ax7, cbar_kws={'shrink': 0.8}, square=True)
    ax7.set_title('Correlation Heatmap (Top 10 Features + Outcome)', fontsize=12, fontweight='bold')

# 7.8: Success Rate by Transfer Type
ax8 = fig.add_subplot(gs[2, 2])
if 'Transfer_Type' in df.columns:
    transfer_success = df.groupby('Transfer_Type').agg({
        'Live_Birth_Success': ['mean', 'count']
    })
    
    ax8.bar(range(len(transfer_success)), transfer_success[('Live_Birth_Success', 'mean')], 
            color='skyblue', alpha=0.7)
    ax8.set_xticks(range(len(transfer_success)))
    ax8.set_xticklabels(transfer_success.index, rotation=45)
    ax8.set_title('Success Rate: Fresh vs Frozen', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Success Rate')
    ax8.set_ylim([0, 1])

plt.suptitle('IVF Dataset - Exploratory Data Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.savefig('ivf_kaggle_exploration.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualizations saved to 'ivf_kaggle_exploration.png'")

# =============================================================================
# STEP 8: DATA QUALITY SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("STEP 8: DATA QUALITY SUMMARY")
print("=" * 70)

print(f"""
DATASET OVERVIEW:
✓ Total Patients: {len(df):,}
✓ Total Features: {len(df.columns)}
✓ Numerical Features: {len(numeric_features)}
✓ Categorical Features: {len(categorical_features)}

TARGET VARIABLE:
✓ Success Rate: {df['Live_Birth_Success'].mean()*100:.1f}%
✓ Total Successes: {df['Live_Birth_Success'].sum()}
✓ Total Failures: {len(df) - df['Live_Birth_Success'].sum()}

DATA QUALITY:
{'✓ No missing values' if missing_df.empty else f'⚠ {len(missing_df)} features with missing data'}
{'✓ No duplicates' if duplicates == 0 else f'⚠ {duplicates} duplicate rows'}

KEY INSIGHTS FROM CORRELATION:
""")

if 'Live_Birth_Success' in df.columns:
    print(f"Most Positive Factors (Top 3):")
    for feat, corr in correlation_with_outcome.head(11)[1:4].items():
        print(f"  • {feat}: {corr:.3f}")
    
    print(f"\nMost Negative Factors (Top 3):")
    for feat, corr in correlation_with_outcome.tail(3).items():
        print(f"  • {feat}: {corr:.3f}")

# =============================================================================
# STEP 9: PREPARE FOR MODELING
# =============================================================================
print("\n" + "=" * 70)
print("STEP 9: FEATURE SELECTION FOR MODELING")
print("=" * 70)

"""
Based on exploration, now identify the most promising features for our model
"""

# Recommend features based on correlation and domain knowledge
recommended_features = []

# Always include these clinical factors
clinical_features = ['Age', 'BMI', 'AMH_Level', 'FSH_Level', 'AFC_Count', 
                     'Endometrial_Thickness_mm', 'Number_of_Embryos_Transferred']

for feat in clinical_features:
    if feat in df.columns:
        recommended_features.append(feat)

# Add highly correlated numeric features
if 'Live_Birth_Success' in df.columns:
    high_corr_features = correlation_with_outcome.abs().sort_values(ascending=False).head(15).index[1:].tolist()
    for feat in high_corr_features:
        if feat not in recommended_features and feat in numeric_features:
            recommended_features.append(feat)

# Add important categorical features
important_categorical = ['Diagnosis_Type', 'Transfer_Type', 'ICSI_or_IVF', 
                         'Embryo_Grade', 'Medication_Protocol']

for feat in important_categorical:
    if feat in df.columns:
        recommended_features.append(feat)

print(f"\nRECOMMENDED FEATURES FOR MODELING ({len(recommended_features)}):")
print("\nNumerical Features:")
for feat in recommended_features:
    if feat in numeric_features:
        if feat in correlation_with_outcome.index:
            corr = correlation_with_outcome[feat]
            print(f"  • {feat:35} (corr: {corr:+.3f})")
        else:
            print(f"  • {feat}")

print("\nCategorical Features (will need encoding):")
for feat in recommended_features:
    if feat in categorical_features:
        unique = df[feat].nunique()
        print(f"  • {feat:35} ({unique} categories)")

# =============================================================================
# STEP 10: SAVE EXPLORATION REPORT
# =============================================================================
print("\n" + "=" * 70)
print("STEP 10: SAVING EXPLORATION REPORT")
print("=" * 70)

# Create a summary report
report = f"""
IVF SUCCESS PREDICTION - DATA EXPLORATION REPORT
{'='*70}

DATASET SOURCE: Kaggle IVF Dataset
DATE: {pd.Timestamp.now().strftime('%Y-%m-%d')}

1. DATASET OVERVIEW
   - Total patients: {len(df):,}
   - Total features: {len(df.columns)}
   - Success rate: {df['Live_Birth_Success'].mean()*100:.2f}%

2. DATA QUALITY
   - Missing values: {len(missing_df)} features affected
   - Duplicate rows: {duplicates}
   
3. TOP PREDICTIVE FEATURES (by correlation)
"""

if 'Live_Birth_Success' in df.columns:
    for i, (feat, corr) in enumerate(correlation_with_outcome.head(11)[1:6].items(), 1):
        report += f"   {i}. {feat}: {corr:+.3f}\n"

report += f"""

4. RECOMMENDED FEATURES FOR MODELING: {len(recommended_features)}

5. NEXT STEPS
   ✓ Handle categorical variables (encoding)
   ✓ Feature scaling/normalization
   ✓ Train/test split (stratified)
   ✓ Build baseline model
   ✓ Build neural network
   
{'='*70}
"""

with open('ivf_exploration_report.txt', 'w') as f:
    f.write(report)

print("\n✓ Exploration report saved to 'ivf_exploration_report.txt'")

# Save the recommended features list
recommended_features_df = pd.DataFrame({
    'feature': recommended_features,
    'type': ['numeric' if f in numeric_features else 'categorical' for f in recommended_features]
})
recommended_features_df.to_csv('recommended_features.csv', index=False)
print("✓ Recommended features saved to 'recommended_features.csv'")
print(f"""
    KEY FINDINGS:
    • Dataset has {len(df):,} patients with {df['Live_Birth_Success'].mean()*100:.1f}% success rate
    • {'Class imbalance detected - will use balanced weights' if outcome_counts.max() / outcome_counts.min() > 2 else 'Classes are balanced'}
    • Top predictor appears to be: {correlation_with_outcome.head(2).index[1]}
    • {len(recommended_features)} features selected for modeling
      """
      )