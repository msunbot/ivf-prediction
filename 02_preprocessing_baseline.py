""" 
IVF Success Prediction - Phase 2: Data Preprocessing & Baseline Model
Working with Kaggle dataset 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STEP 1: LOAD THE DATA
# =============================================================================
print("=" * 70)
print("PHASE 2: DATA PREPROCESSING & BASELINE MODEL")
print("=" * 70)

df = pd.read_csv("kaggle_data_set.csv")
print(f"\n✓ Loaded {len(df)} patient records")
print(f"  Features: {len(df.columns)}")
print(f"  Success rate: {df['Live_Birth_Success'].mean()*100:.2f}%\n")

# =============================================================================
# STEP 2: FEATURE SELECTION
# =============================================================================
print("=" * 70)
print("STEP 1: FEATURE SELECTION")
print("-" * 70)

""" 
Based on Phase 1 exploration, we select features that: 
1. Have medical relevance to IVF outcomes
2. Show correlation with success
3. Have complete or mostly complete data
4. Are available at prediction time

And exclude features known AFTER the cycle: 
- implantation success
- cycle cost
"""

# Define our feature set
numerical_features = [
    # Patient Demographics
    "Age", 
    "BMI", 
    "Years_of_Infertility", 

    # Hormonal Markers (critical for IVF)
    "AMH_Level", 
    "FSH_Level", 
    "LH_Level", 
    "Estrogen_E2_Level", 
    "Progesterone_P4_Level", 

    # Uterine Factors
    "Endometrial_Thickness_mm", 
    "AFC_Count", 

    # Metabolic Markers
    'Thyroid_TSH',
    'Insulin_Level',
    
    # Treatment Details
    'Number_of_IVF_Cycles',
    'Pregnancy_History',
    'Number_of_Embryos_Transferred',
    
    # Lifestyle Factors
    'Diet_Quality_Score',
    'Yoga_Sessions_Per_Week',
    'Stress_Level',
    'Physical_Activity_Hours_Per_Week',
    'Sleep_Duration_Hours',
    
    # Male Factors
    'Sperm_Count',
    'Sperm_Motility (%)',
    'Sperm_Morphology (%)',
    'Sperm_DNA_Fragmentation (%)',
]

categorical_features = [
    'Diagnosis_Type',
    'Medication_Protocol',
    'Luteal_Support_Given',
    'PGT',
    'Transfer_Type',
    'Day_of_Transfer',
    'ICSI_or_IVF',
    'Assisted_Hatching_Used',
    'Smoking_Status',
    'Alcohol_Consumption',
    'Exposure_to_Environmental_Toxins',
    'Occupation_Type',
    'Male_Infertility_Diagnosis',
]

# Check which features are actually in the dataset
available_numerical = [f for f in numerical_features if f in df.columns]
available_categorical = [f for f in categorical_features if f in df.columns]

print(f"\n✓ Selected {len(available_numerical)} numerical features")
print(f"✓ Selected {len(available_categorical)} categorical features")
print(f"✓ Total features for modeling: {len(available_numerical) + len(available_categorical)}")

# =============================================================================
# STEP 3: HANDLE MISSING VALUES (DO THIS FIRST!)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: HANDLING MISSING VALUES")
print("-" * 70)

"""
Real data often has missing values. Strategies:
1. Drop rows (if < 5% missing and dataset is large)
2. Fill with mean/median (for numerical)
3. Fill with mode (for categorical)
4. Create 'missing' indicator features (advanced)

For this dataset, we'll use strategy 2 & 3
"""

# Create a copy for processing
df_processed = df.copy()

# Check missing values
missing_before = df_processed[available_numerical + available_categorical].isnull().sum().sum()
print(f"\nTotal missing values before: {missing_before}")

if missing_before > 0:
    print("\nFilling missing values...")
    
    # Fill numerical with median (more robust to outliers than mean)
    for col in available_numerical:
        if df_processed[col].isnull().any():
            median_val = df_processed[col].median()
            n_missing = df_processed[col].isnull().sum()
            df_processed[col].fillna(median_val, inplace=True)
            print(f"  • {col}: filled {n_missing} missing with median ({median_val:.2f})")
    
    # Fill categorical with mode (most common value) OR 'Unknown'
    for col in available_categorical:
        if df_processed[col].isnull().any():
            n_missing = df_processed[col].isnull().sum()
        
            # Special handling for diagnosis/medical fields
            # Missing likely means "no diagnosis" or "not applicable"
            if 'Diagnosis' in col or 'Infertility' in col:
                fill_val = 'None'  # or 'Unknown' or 'Not_Applicable'
                print(f"  • {col}: filled {n_missing} missing with '{fill_val}' (no diagnosis)")
            else:
                # For other categorical fields, use mode
                if df_processed[col].notna().any():
                    fill_val = df_processed[col].mode()[0]
                    print(f"  • {col}: filled {n_missing} missing with mode ('{fill_val}')")
                else:
                    fill_val = 'Unknown'
                    print(f"  • {col}: filled {n_missing} missing with 'Unknown'")
            
            df_processed[col].fillna(fill_val, inplace=True)
    
    missing_after = df_processed[available_numerical + available_categorical].isnull().sum().sum()
    print(f"\n✓ Missing values reduced from {missing_before} to {missing_after}")
else:
    print("✓ No missing values in selected features!")

# =============================================================================
# STEP 4: ENCODE CATEGORICAL VARIABLES (NOW SAFE - NO NaN!)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: ENCODING CATEGORICAL VARIABLES")
print("-" * 70)

"""
WHY? ML models need numbers, not text.

METHODS:
1. Label Encoding: A→0, B→1, C→2 (for tree-based or ordinal)
2. One-Hot Encoding: Creates binary columns (for linear models/neural nets)

We'll use Label Encoding for simplicity and compatibility with both model types
"""

# Store encoders for later use
encoders = {}

print("\nEncoding categorical features:")
for col in available_categorical:
    le = LabelEncoder()
    # Now safe because we filled missing values above!
    df_processed[col + '_encoded'] = le.fit_transform(df_processed[col].astype(str))
    encoders[col] = le
    
    # Show mapping for first few categories
    unique_vals = df_processed[col].unique()[:5]
    print(f"\n  {col}:")
    for val in unique_vals:
        encoded = le.transform([str(val)])[0]
        print(f"    '{val}' → {encoded}")
    if len(df_processed[col].unique()) > 5:
        print(f"    ... and {len(df_processed[col].unique()) - 5} more categories")

# Create list of encoded feature names
encoded_features = [col + '_encoded' for col in available_categorical]

# =============================================================================
# STEP 5: PREPARE FEATURE MATRIX AND TARGET
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: PREPARING FEATURE MATRIX (X) AND TARGET (y)")
print("-" * 70)

# Combine numerical and encoded categorical features
all_features = available_numerical + encoded_features

X = df_processed[all_features]
y = df_processed['Live_Birth_Success']

print(f"\n✓ Feature matrix (X): {X.shape}")
print(f"✓ Target vector (y): {y.shape}")
print(f"\n  Total features: {len(all_features)}")
print(f"  - Numerical: {len(available_numerical)}")
print(f"  - Categorical (encoded): {len(encoded_features)}")

# =============================================================================
# STEP 6: TRAIN/TEST SPLIT
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: TRAIN/TEST SPLIT (80/20)")
print("-" * 70)

"""
CRITICAL: Always split BEFORE any processing that uses statistics
(like scaling). Otherwise you leak information from test → train.

stratify=y ensures both sets have same success/failure ratio
"""

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, # ensure reproducibility, guarantee that data splitting process if consistent 
    stratify=y
)

print(f"\nTraining set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test set:     {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# Check class distribution
train_success_rate = y_train.mean()
test_success_rate = y_test.mean()
print(f"\nSuccess rate in training: {train_success_rate*100:.2f}%")
print(f"Success rate in test:     {test_success_rate*100:.2f}%")
print(f"Difference: {abs(train_success_rate - test_success_rate)*100:.2f}% (should be small)")

# =============================================================================
# STEP 7: FEATURE SCALING
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: FEATURE SCALING (STANDARDIZATION)")
print("-" * 70)

"""
WHY SCALE?
Features have wildly different ranges:
- Age: 20-50
- AMH: 0-15
- Sperm_Count: 0-200+

Without scaling:
- Larger features dominate
- Gradient descent converges slowly
- Model coefficients hard to interpret

StandardScaler: (x - mean) / std
Result: mean=0, std=1 for each feature
"""

scaler = StandardScaler()

# Fit on training data 
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Use training statistics

print("\nBefore scaling (Age feature):")
print(f"  Mean: {X_train['Age'].mean():.2f}")
print(f"  Std:  {X_train['Age'].std():.2f}")
print(f"  Range: [{X_train['Age'].min():.1f}, {X_train['Age'].max():.1f}]")

age_idx = all_features.index('Age')
print("\nAfter scaling (Age feature):")
print(f"  Mean: {X_train_scaled[:, age_idx].mean():.4f}")
print(f"  Std:  {X_train_scaled[:, age_idx].std():.4f}")

# =============================================================================
# STEP 8: BUILD BASELINE MODEL (LOGISTIC REGRESSION)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: TRAINING BASELINE MODEL (LOGISTIC REGRESSION)")
print("-" * 70)

"""
WHY LOGISTIC REGRESSION FIRST?
✓ Fast to train (seconds, not minutes)
✓ Interpretable (can see which features matter)
✓ Strong baseline (often hard to beat on tabular data)
✓ Handles binary outcomes naturally

class_weight='balanced' adjusts for class imbalance by weighting:
  weight = n_samples / (n_classes * n_samples_in_class)
"""

# Check class balance
class_balance = y_train.value_counts()
print(f"\nClass distribution in training:")
print(f"  Failure (0): {class_balance[0]} ({class_balance[0]/len(y_train)*100:.1f}%)")
print(f"  Success (1): {class_balance[1]} ({class_balance[1]/len(y_train)*100:.1f}%)")

# Train model
model_lr = LogisticRegression(
    max_iter=1000, 
    random_state=42,
    class_weight='balanced', # handle imbalance
    C=1.0 # regularization strength (lower = more regularization)
)

print("\nTraining logistic regression")
model_lr.fit(X_train_scaled, y_train)
print("✓ Model trained!")

# =============================================================================
# STEP 9: MAKE PREDICTIONS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 8: MAKING PREDICTIONS")
print("-" * 70)

# Get predictions
y_pred_train = model_lr.predict(X_train_scaled)
y_pred_test = model_lr.predict(X_test_scaled)

# Get probabilities
y_pred_proba_train = model_lr.predict_proba(X_train_scaled)[:, 1]
y_pred_proba_test = model_lr.predict_proba(X_test_scaled)[:, 1]

print("\nExample predictions (first 10 test patients):")
print("=" * 60)
print("Actual | Predicted | Probability | Interpretation")
print("-" * 60)
for i in range(min(10, len(y_test))):
    actual = y_test.iloc[i]
    pred = y_pred_test[i]
    prob = y_pred_proba_test[i]
    status = "✓ Correct" if actual == pred else "✗ Wrong"
    interp = "High confidence" if abs(prob - 0.5) > 0.3 else "Uncertain"
    print(f"  {actual}    |     {pred}     |   {prob:.3f}   | {interp:15} {status}")

# =============================================================================
# STEP 10: EVALUATE MODEL
# =============================================================================
print("\n" + "=" * 70)
print("STEP 9: MODEL EVALUATION")
print("-" * 70)

"""
KEY METRICS:

ACCURACY: (TP + TN) / Total
- Good for balanced datasets
- Can be misleading with imbalance

PRECISION: TP / (TP + FP)
- "Of predicted successes, how many were right?"
- High precision = fewer false hopes

RECALL: TP / (TP + FN)
- "Of actual successes, how many did we catch?"
- High recall = don't miss viable cases

F1-SCORE: 2 * (Precision * Recall) / (Precision + Recall)
- Harmonic mean, good for imbalanced data

AUC-ROC: Area under ROC curve
- Measures discrimination ability
- 0.5 = random, 1.0 = perfect
- >0.7 = acceptable, >0.8 = good, >0.9 = excellent
"""

def evaluate_model(y_true, y_pred, y_pred_proba, dataset_name):
    print(f"\n{dataset_name} SET PERFORMANCE:")
    print("=" * 60)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f} (when we predict success, we're right {prec*100:.1f}% of the time)")
    print(f"Recall:    {rec:.4f} (we catch {rec*100:.1f}% of actual successes)")
    print(f"F1-Score:  {f1:.4f} (harmonic mean of precision & recall)")
    print(f"AUC-ROC:   {auc:.4f} (discrimination ability)")
    
    # Interpretation
    if auc >= 0.9:
        interp = "EXCELLENT"
    elif auc >= 0.8:
        interp = "GOOD"
    elif auc >= 0.7:
        interp = "ACCEPTABLE"
    else:
        interp = "NEEDS IMPROVEMENT"
    print(f"\nOverall: {interp} model")
    
    return acc, prec, rec, f1, auc

# Evaluate
train_metrics = evaluate_model(y_train, y_pred_train, y_pred_proba_train, "TRAINING")
test_metrics = evaluate_model(y_test, y_pred_test, y_pred_proba_test, "TEST")

# Overfitting check
print("\n" + "=" * 70)
print("OVERFITTING CHECK")
print("-" * 70)
train_acc, test_acc = train_metrics[0], test_metrics[0]
gap = train_acc - test_acc

print(f"Training accuracy: {train_acc:.4f}")
print(f"Test accuracy:     {test_acc:.4f}")
print(f"Gap:               {gap:.4f} ({gap*100:.2f} percentage points)")

if gap < 0.02:
    print("✓ EXCELLENT: Model generalizes very well")
elif gap < 0.05:
    print("✓ GOOD: Acceptable generalization")
elif gap < 0.10:
    print("⚠ ACCEPTABLE: Slight overfitting")
else:
    print("✗ WARNING: Significant overfitting detected")

# =============================================================================
# STEP 11: CONFUSION MATRIX
# =============================================================================
print("\n" + "=" * 70)
print("STEP 10: CONFUSION MATRIX ANALYSIS")
print("-" * 70)

cm = confusion_matrix(y_test, y_pred_test)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print("                    PREDICTED")
print("                 Fail    Success")
print(f"ACTUAL  Fail     {tn:4d}      {fp:4d}    (Specificity: {tn/(tn+fp)*100:.1f}%)")
print(f"      Success    {fn:4d}      {tp:4d}    (Sensitivity: {tp/(tp+fn)*100:.1f}%)")

print("\nInterpretation:")
print(f"  True Negatives (TN):  {tn:4d} - Correctly predicted failures")
print(f"  True Positives (TP):  {tp:4d} - Correctly predicted successes ✓")
print(f"  False Positives (FP): {fp:4d} - False hope (predicted success, but failed)")
print(f"  False Negatives (FN): {fn:4d} - Missed opportunities (predicted failure, but succeeded)")

print(f"\nClinical Impact:")
print(f"  • False Positive Rate: {fp/(fp+tn)*100:.1f}% (giving false hope)")
print(f"  • False Negative Rate: {fn/(fn+tp)*100:.1f}% (missing viable cases)")

# =============================================================================
# STEP 12: FEATURE IMPORTANCE
# =============================================================================
print("\n" + "=" * 70)
print("STEP 11: FEATURE IMPORTANCE")
print("-" * 70)

"""
Logistic Regression coefficients tell us:
- POSITIVE: Feature increases probability of success
- NEGATIVE: Feature decreases probability of success
- MAGNITUDE: How strong the effect is (after scaling)
"""

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': all_features,
    'Coefficient': model_lr.coef_[0],
    'Abs_Coefficient': np.abs(model_lr.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("\nTOP 10 MOST IMPORTANT FEATURES:")
print("=" * 70)
print(f"{'Feature':<35} {'Coefficient':>12} {'Impact':>10}")
print("-" * 70)
for idx, row in feature_importance.head(10).iterrows():
    impact = "↑ Increases" if row['Coefficient'] > 0 else "↓ Decreases"
    print(f"{row['Feature']:<35} {row['Coefficient']:>12.4f} {impact:>10}")

# =============================================================================
# STEP 13: VISUALIZATIONS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 12: CREATING VISUALIZATIONS")
print("-" * 70)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Baseline Model (Logistic Regression) - Evaluation', fontsize=16, fontweight='bold')

# 13.1: Confusion Matrix Heatmap
ax1 = axes[0, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['Fail', 'Success'], 
            yticklabels=['Fail', 'Success'],
            cbar_kws={'label': 'Count'})
ax1.set_title('Confusion Matrix', fontweight='bold')
ax1.set_ylabel('Actual')
ax1.set_xlabel('Predicted')

# 13.2: ROC Curve
ax2 = axes[0, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_test)
ax2.plot(fpr, tpr, linewidth=2, label=f'Model (AUC = {test_metrics[4]:.3f})', color='blue')
ax2.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
ax2.set_title('ROC Curve', fontweight='bold')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate (Recall)')
ax2.legend()
ax2.grid(alpha=0.3)

# 13.3: Feature Importance (Top 15)
ax3 = axes[0, 2]
top_features = feature_importance.head(15)
colors = ['green' if x > 0 else 'red' for x in top_features['Coefficient']]
ax3.barh(range(len(top_features)), top_features['Coefficient'], color=colors, alpha=0.7)
ax3.set_yticks(range(len(top_features)))
ax3.set_yticklabels(top_features['Feature'], fontsize=8)
ax3.set_title('Feature Importance (Top 15)', fontweight='bold')
ax3.set_xlabel('Coefficient')
ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
ax3.grid(axis='x', alpha=0.3)

# 13.4: Prediction Distribution
ax4 = axes[1, 0]
ax4.hist(y_pred_proba_test[y_test==0], bins=30, alpha=0.6, 
         label='Actual Failures', color='red', edgecolor='black')
ax4.hist(y_pred_proba_test[y_test==1], bins=30, alpha=0.6,
         label='Actual Successes', color='green', edgecolor='black')
ax4.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
ax4.set_title('Predicted Probability Distribution', fontweight='bold')
ax4.set_xlabel('Predicted Probability of Success')
ax4.set_ylabel('Count')
ax4.legend()

# 13.5: Performance Metrics Comparison
ax5 = axes[1, 1]
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
train_vals = list(train_metrics)
test_vals = list(test_metrics)
x = np.arange(len(metrics_names))
width = 0.35
ax5.bar(x - width/2, train_vals, width, label='Training', alpha=0.8, color='skyblue')
ax5.bar(x + width/2, test_vals, width, label='Test', alpha=0.8, color='coral')
ax5.set_ylabel('Score')
ax5.set_title('Training vs Test Performance', fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(metrics_names, rotation=45, ha='right')
ax5.legend()
ax5.set_ylim([0, 1])
ax5.grid(axis='y', alpha=0.3)

# 13.6: Calibration-style plot
ax6 = axes[1, 2]
prob_bins = np.linspace(0, 1, 11)
bin_centers = (prob_bins[:-1] + prob_bins[1:]) / 2
bin_true = []
for i in range(len(prob_bins)-1):
    mask = (y_pred_proba_test >= prob_bins[i]) & (y_pred_proba_test < prob_bins[i+1])
    if mask.sum() > 0:
        bin_true.append(y_test[mask].mean())
    else:
        bin_true.append(np.nan)

ax6.plot(bin_centers, bin_true, 'o-', linewidth=2, markersize=8, label='Model', color='blue')
ax6.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=1)
ax6.set_title('Prediction Calibration', fontweight='bold')
ax6.set_xlabel('Predicted Probability')
ax6.set_ylabel('Observed Frequency')
ax6.legend()
ax6.grid(alpha=0.3)
ax6.set_xlim([0, 1])
ax6.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('baseline_model_evaluation_kaggle.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualizations saved to 'baseline_model_evaluation_kaggle.png'")

# =============================================================================
# STEP 14: SAVE RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 13: SAVING RESULTS")
print("-" * 70)

# Save model performance
results_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
    'Training': train_metrics,
    'Test': test_metrics
})
results_df.to_csv('baseline_model_results.csv', index=False)
print("✓ Results saved to 'baseline_model_results.csv'")

# Save feature importance
feature_importance.to_csv('feature_importance.csv', index=False)
print("✓ Feature importance saved to 'feature_importance.csv'")

# Save the scaler and model (for future use)
import joblib
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model_lr, 'logistic_regression_model.pkl')
joblib.dump(encoders, 'label_encoders.pkl')
print("✓ Model artifacts saved (scaler.pkl, logistic_regression_model.pkl, label_encoders.pkl)")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 2 COMPLETE! 🎉")
print("=" * 70)
print(f"""
BASELINE MODEL SUMMARY:
{'='*70}

MODEL: Logistic Regression with Balanced Class Weights
FEATURES: {len(all_features)} total ({len(available_numerical)} numerical, {len(encoded_features)} categorical)

PERFORMANCE ON TEST SET:
  • Accuracy:  {test_metrics[0]:.2%}
  • Precision: {test_metrics[1]:.4f}
  • Recall:    {test_metrics[2]:.4f}
  • F1-Score:  {test_metrics[3]:.4f}
  • AUC-ROC:   {test_metrics[4]:.4f}

TOP 3 PREDICTIVE FEATURES:
  1. {feature_importance.iloc[0]['Feature']} ({feature_importance.iloc[0]['Coefficient']:+.3f})
  2. {feature_importance.iloc[1]['Feature']} ({feature_importance.iloc[1]['Coefficient']:+.3f})
  3. {feature_importance.iloc[2]['Feature']} ({feature_importance.iloc[2]['Coefficient']:+.3f})

GENERALIZATION: {'Excellent' if gap < 0.02 else 'Good' if gap < 0.05 else 'Acceptable'}
  (Train-test accuracy gap: {gap:.4f})

FILES CREATED:
  ✓ baseline_model_evaluation_kaggle.png
  ✓ baseline_model_results.csv
  ✓ feature_importance.csv
  ✓ Model artifacts (scaler.pkl, model.pkl, encoders.pkl)

NEXT STEPS:
  Ready for Phase 3: Neural Network Implementation!
  The neural network should aim to beat AUC = {test_metrics[4]:.3f}
  
  Command: python phase3_neural_network.py
{'='*70}
""")
import numpy as np
from pathlib import Path

Path(".").mkdir(exist_ok=True)

np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

# (Optional if you also have a validation split)
# np.save("X_val.npy", X_val_arr)
# np.save("y_val.npy", y_val)