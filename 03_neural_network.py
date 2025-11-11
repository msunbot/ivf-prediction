"""
IVF Success Prediction - Phase 3: Neural Network Implementation
Building a deep learning model to beat the baseline!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, roc_curve, confusion_matrix)
import joblib
import warnings
warnings.filterwarnings('ignore')

from models.model_configs import get_model, get_config

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# =============================================================================
# STEP 1: LOAD AND PREPARE DATA (Same as Phase 2)
# =============================================================================
print("=" * 70)
print("PHASE 3: NEURAL NETWORK IMPLEMENTATION")
print("=" * 70)

df = pd.read_csv('data/kaggle_data_set.csv')
print(f"\n✓ Dataset loaded: {len(df)} patients")

# Feature lists (same as Phase 2)
numerical_features = [
    'Age', 'BMI', 'Years_of_Infertility',
    'AMH_Level', 'FSH_Level', 'LH_Level', 'Estrogen_E2_Level', 'Progesterone_P4_Level',
    'Endometrial_Thickness_mm', 'AFC_Count', 'Thyroid_TSH', 'Insulin_Level',
    'Number_of_IVF_Cycles', 'Pregnancy_History', 'Number_of_Embryos_Transferred',
    'Diet_Quality_Score', 'Yoga_Sessions_Per_Week', 'Stress_Level',
    'Physical_Activity_Hours_Per_Week', 'Sleep_Duration_Hours',
    'Sperm_Count', 'Sperm_Motility (%)', 'Sperm_Morphology (%)', 'Sperm_DNA_Fragmentation (%)',
]

categorical_features = [
    'Diagnosis_Type', 'Medication_Protocol', 'Luteal_Support_Given', 'PGT',
    'Transfer_Type', 'Day_of_Transfer', 'ICSI_or_IVF', 'Assisted_Hatching_Used',
    'Smoking_Status', 'Alcohol_Consumption', 'Exposure_to_Environmental_Toxins',
    'Occupation_Type', 'Male_Infertility_Diagnosis',
]

available_numerical = [f for f in numerical_features if f in df.columns]
available_categorical = [f for f in categorical_features if f in df.columns]

# Handle missing values
df_processed = df.copy()
for col in available_numerical:
    if df_processed[col].isnull().any():
        df_processed[col].fillna(df_processed[col].median(), inplace=True)

for col in available_categorical:
    if df_processed[col].isnull().any():
        df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)

# Encode categorical variables
for col in available_categorical:
    le = LabelEncoder()
    df_processed[col + '_encoded'] = le.fit_transform(df_processed[col].astype(str))

encoded_features = [col + '_encoded' for col in available_categorical]
all_features = available_numerical + encoded_features

# Prepare X and y
X = df_processed[all_features].values
y = df_processed['Live_Birth_Success'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Calculate class weights
class_counts = np.bincount(y_train)
class_weights = len(y_train) / (len(class_counts) * class_counts)
pos_weight = torch.FloatTensor([class_weights[1] / class_weights[0]])

print(f"\nClass distribution:")
print(f"  Failures: {class_counts[0]} ({class_counts[0]/len(y_train)*100:.1f}%)")
print(f"  Successes: {class_counts[1]} ({class_counts[1]/len(y_train)*100:.1f}%)")
print(f"  Positive class weight: {pos_weight.item():.3f}")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Features prepared: {len(all_features)} features")
print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Test samples: {len(X_test)}")

# Load baseline results for comparison
try:
    baseline_results = pd.read_csv('baseline_model_results.csv')
    baseline_auc = baseline_results[baseline_results['Metric'] == 'AUC-ROC']['Test'].values[0]
    print(f"\n🎯 Goal: Beat baseline AUC of {baseline_auc:.4f}")
except:
    baseline_auc = None
    print("\n📊 No baseline results found - training from scratch!")

# =============================================================================
# STEP 2: CONVERT TO PYTORCH TENSORS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1: CONVERTING TO PYTORCH TENSORS")
print("-" * 70)

X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

print(f"\n✓ Tensors created:")
print(f"  X_train: {X_train_tensor.shape}")
print(f"  y_train: {y_train_tensor.shape}")

# =============================================================================
# STEP 3: CREATE DATA LOADERS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: CREATING DATA LOADERS")
print("-" * 70)

BATCH_SIZE = 32

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\n✓ Data loaders created")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Training batches: {len(train_loader)}")
print(f"  Test batches: {len(test_loader)}")

# =============================================================================
# STEP 4: DEFINE NEURAL NETWORK ARCHITECTURE
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: LOADING MODEL CONFIGURATION")
print("-" * 70)

"""
ARCHITECTURE FOR IVF PREDICTION:

We have {len(all_features)} input features - this is a moderately sized feature set.

Network Design: 
- Input: {len(all_features)} features
- Hidden Layer 1: 128 neurons (wide layer to capture complex patterns)
- Hidden Layer 2: 64 neurons (refinement)
- Hidden Layer 3: 32 neurons (focused representation)
- Output: 1 neuron (success probability)

KEY COMPONENTS: 
- BatchNorm: Stabilizes training, reduces internal covariate shift
- ReLU: Non-Linear activation (allows learning complex patterns)
- Dropout: Regularization to prevent overfitting (rate=0.3)
- Sigmoid: Output activation (converts to probability 0-1)
"""

# Choose which model to train
MODEL_TYPE = "balanced" # Options: "balanced", "clinical", "production"

# Get model and config
input_size = len(all_features)
model = get_model(MODEL_TYPE, input_size)
config = get_config(MODEL_TYPE)

print(f"\n✓ Using: {config['name']}")
print(f"  Description: {config['description']}")
print(f"  Dropout rate: {config['dropout']}")
print(f"\nModel Architecture:")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nParameters: {total_params:,}")

# =============================================================================
# STEP 5: LOSS FUNCTION & OPTIMIZER
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: LOSS FUNCTION & OPTIMIZER")
print("-" * 70)

"""
LOSS: BCELoss (Binary Cross-Entropy)
- Standard for binary classification 
- Measures how far predictions are from truth 

OPTIMIZER: Adam
- Adaptive learning rate for each parameter
- Combines momentum + RMSprop
- Learning rate: 0.001 (standard starting point)
- Weight decay: 1e-5 (L2 regularization to prevent overfitting)
"""

# Load training parameters from config 
criterion = nn.BCELoss() # output a scalar (a probability between 0 and 1)

optimizer = optim.Adam(
    model.parameters(), 
    lr=config["learning_rate"],  
    weight_decay=config["weight_decay"]) 

# Learning rate scheduler (reduces LR if validation loss plateaus)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, 
    patience=10, min_lr=1e-6 
)

NUM_EPOCHS = config["max_epochs"]   
PATIENCE = config["patience"]       
BATCH_SIZE = config["batch_size"]

print(f"\n✓ Training setup from {MODEL_TYPE} config:")
print(f"  Optimizer: Adam")
print(f"  Learning rate: {config['learning_rate']}")
print(f"  Weight decay: {config['weight_decay']}")
print(f"  Max epochs: {NUM_EPOCHS}")
print(f"  Patience: {PATIENCE}")

# =============================================================================
# STEP 6: TRAINING LOOP
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: TRAINING THE NEURAL NETWORK")
print("-" * 70)



# Track training history 
history = { 
    "train_loss" : [],
    "train_acc"  : [], 
    "val_loss"   : [],
    "val_acc"    : [],
    "learning_rate" : []
}

best_val_loss = float("inf") # set a variable with infinitely large value
best_val_auc = 0
patience_counter = 0 

print(f"\nTraining configuration:")
print(f"  Max epochs: {NUM_EPOCHS}")
print(f"  Early stopping patience: {PATIENCE}")
print(f"  Batch size: {BATCH_SIZE}")
print("\n" + "=" * 70)
print("Starting training...")
print("-" * 70)

for epoch in range(NUM_EPOCHS):
    # ==================== TRAINING PHASE ====================
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        train_loss += loss.item() # loss is a pytorch tensor, .item() extracts the python number 
        predictions = (outputs >= 0.5).float()
        train_correct += (predictions == batch_y).sum().item()
        train_total += batch_y.size(0) 

    avg_train_loss = train_loss / len(train_loader)
    train_accuracy = train_correct / train_total

    # ==================== VALIDATION PHASE ====================
    model.eval()
    val_loss = 0.0 
    val_correct = 0
    val_total = 0 
    all_val_probs = []
    all_val_labels = []

    with torch.no_grad(): 
        for batch_X, batch_y in test_loader: 
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            val_loss += loss.item()
            predictions = (outputs >= 0.5).float()
            val_correct += (predictions == batch_y).sum().item()
            val_total += batch_y.size(0) 

            all_val_probs.extend(outputs.numpy())
            all_val_labels.extend(batch_y.numpy())

    avg_val_loss = val_loss / len(test_loader)
    val_accuracy = val_correct / val_total
    val_auc = roc_auc_score(all_val_labels, all_val_probs)

    # Update learning rate
    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    # Store history
    history['train_loss'].append(avg_train_loss)
    history['train_acc'].append(train_accuracy)
    history['val_loss'].append(avg_val_loss)
    history['val_acc'].append(val_accuracy)
    history['learning_rate'].append(current_lr)

    # Print progress
    if (epoch + 1) % 10 == 0 or epoch < 5: 
        print(f"Epoch [{epoch+1:3d}/{NUM_EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f} | "
              f"Val AUC: {val_auc:.4f}")
        
    # Early stopping & model saving
    if val_auc > best_val_auc: 
        best_val_auc = val_auc
        best_val_loss = avg_val_loss
        pateince_counter = 0 
        torch.save(model.state_dict(), config['save_path'])
        print(f"   -> Model saved to {config['save_path']}")
        if epoch > 10: # don't print the very early epochs
            print(f"  -> New best AUC: {best_val_auc:.4f} (model saved)")
    else: 
        patience_counter += 1
        if patience_counter >= PATIENCE: 
            print(f"\n✓ Early stopping triggered at epoch {epoch+1}")
            print(f"  Best validation AUC: {best_val_auc:.4f}")
            break

# Load best model
model.load_state_dict(torch.load(config['save_path']))
print(f"\n✓ Training complete!")
print(f"  Total epochs: {len(history['train_loss'])}")
print(f"  Best validation AUC: {best_val_auc:.4f}")

# =============================================================================
# STEP 7: FINAL EVALUATION
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: FINAL EVALUATION ON TEST SET")
print("-" * 70)

model.eval()
with torch.no_grad():
    # logits = model(X_test_tensor)
    # y_pred_proba = torch.sigmoid(logits).numpy().ravel() Fix #2, now reverted
    y_pred_proba = model(X_test_tensor).numpy().ravel() 
    y_pred = (y_pred_proba >= 0.5).astype(int)

# Calculate all metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n{'='*70}")
print("NEURAL NETWORK FINAL PERFORMANCE")
print(f"{'='*70}")
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"AUC-ROC:   {auc:.4f}")

# Compare with baseline
if baseline_auc is not None:
    improvement = auc - baseline_auc
    pct_improvement = (improvement / baseline_auc) * 100
    print(f"\n{'='*70}")
    print("COMPARISON WITH BASELINE")
    print(f"{'='*70}")
    print(f"Baseline (Logistic Regression) AUC: {baseline_auc:.4f}")
    print(f"Neural Network AUC:                 {auc:.4f}")
    print(f"Improvement:                        {improvement:+.4f} ({pct_improvement:+.2f}%)")
    
    if improvement > 0.02:
        print("\n🎉 SUCCESS! Neural network significantly outperforms baseline!")
    elif improvement > 0:
        print("\n✓ Neural network slightly better than baseline")
    else:
        print("\n⚠ Neural network didn't beat baseline (try hyperparameter tuning)")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n{'='*70}")
print("CONFUSION MATRIX")
print(f"{'='*70}")
print(f"                 PREDICTED")
print(f"              Fail    Success")
print(f"ACTUAL Fail   {tn:4d}      {fp:4d}")
print(f"     Success  {fn:4d}      {tp:4d}")

# =============================================================================
# STEP 8: VISUALIZATIONS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: CREATING VISUALIZATIONS")
print("-" * 70)

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle('Neural Network - Training & Evaluation', fontsize=16, fontweight='bold')

# 8.1: Training Loss
ax1 = axes[0, 0]
ax1.plot(history['train_loss'], label='Training Loss', linewidth=2, alpha=0.8)
ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2, alpha=0.8)
ax1.set_title('Loss Over Epochs', fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(alpha=0.3)

# 8.2: Training Accuracy
ax2 = axes[0, 1]
ax2.plot(history['train_acc'], label='Training Accuracy', linewidth=2, alpha=0.8)
ax2.plot(history['val_acc'], label='Validation Accuracy', linewidth=2, alpha=0.8)
ax2.set_title('Accuracy Over Epochs', fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(alpha=0.3)

# 8.3: Learning Rate Schedule
ax3 = axes[0, 2]
ax3.plot(history['learning_rate'], linewidth=2, color='orange')
ax3.set_title('Learning Rate Schedule', fontweight='bold')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Learning Rate')
ax3.set_yscale('log')
ax3.grid(alpha=0.3)

# 8.4: ROC Curve
ax4 = axes[1, 0]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
ax4.plot(fpr, tpr, linewidth=2, label=f'NN (AUC={auc:.3f})', color='blue')
if baseline_auc is not None:
    ax4.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax4.axhline(y=baseline_auc, color='red', linestyle='--', 
               label=f'Baseline AUC={baseline_auc:.3f}', alpha=0.6)
else:
    ax4.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
ax4.set_title('ROC Curve', fontweight='bold')
ax4.set_xlabel('False Positive Rate')
ax4.set_ylabel('True Positive Rate')
ax4.legend()
ax4.grid(alpha=0.3)

# 8.5: Confusion Matrix
ax5 = axes[1, 1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
           xticklabels=['Fail', 'Success'],
           yticklabels=['Fail', 'Success'])
ax5.set_title('Confusion Matrix', fontweight='bold')
ax5.set_ylabel('Actual')
ax5.set_xlabel('Predicted')

# 8.6: Prediction Distribution
ax6 = axes[1, 2]
ax6.hist(y_pred_proba[y_test==0], bins=30, alpha=0.6, 
        label='Actual Failures', color='red', edgecolor='black')
ax6.hist(y_pred_proba[y_test==1], bins=30, alpha=0.6,
        label='Actual Successes', color='green', edgecolor='black')
ax6.axvline(x=0.5, color='black', linestyle='--', linewidth=2)
ax6.set_title('Predicted Probability Distribution', fontweight='bold')
ax6.set_xlabel('Predicted Probability')
ax6.set_ylabel('Count')
ax6.legend()

# 8.7: Metrics Comparison (if baseline exists)
ax7 = axes[2, 0]
if baseline_auc is not None:
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    baseline_vals = baseline_results['Test'].values
    nn_vals = [accuracy, precision, recall, f1, auc]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax7.bar(x - width/2, baseline_vals, width, label='Baseline', alpha=0.8, color='coral')
    ax7.bar(x + width/2, nn_vals, width, label='Neural Network', alpha=0.8, color='skyblue')
    ax7.set_ylabel('Score')
    ax7.set_title('Model Comparison', fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels(metrics)
    ax7.legend()
    ax7.set_ylim([0, 1])
    ax7.grid(axis='y', alpha=0.3)
else:
    ax7.text(0.5, 0.5, 'No baseline\ncomparison available', 
            ha='center', va='center', fontsize=12)
    ax7.axis('off')

# 8.8: Overfitting Check
ax8 = axes[2, 1]
epochs = range(1, len(history['train_loss']) + 1)
ax8.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
ax8.plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
ax8.set_title('Overfitting Check', fontweight='bold')
ax8.set_xlabel('Epoch')
ax8.set_ylabel('Loss')
ax8.legend()
ax8.grid(alpha=0.3)
gap = abs(history['train_loss'][-1] - history['val_loss'][-1])
ax8.text(0.98, 0.98, f'Final gap: {gap:.4f}', 
        transform=ax8.transAxes, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 8.9: Performance Summary
ax9 = axes[2, 2]
ax9.axis('off')
summary_text = f"""
FINAL RESULTS SUMMARY

Test Set Performance:
• Accuracy:  {accuracy:.2%}
• Precision: {precision:.4f}
• Recall:    {recall:.4f}
• F1-Score:  {f1:.4f}
• AUC-ROC:   {auc:.4f}

Training:
• Epochs: {len(history['train_loss'])}
• Best Val AUC: {best_val_auc:.4f}

Model: {input_size}→128→64→32→1
Parameters: {total_params:,}
"""
ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, 
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.savefig('neural_network_results_kaggle_07_single_hidden_layer.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualizations saved to 'neural_network_results_kaggle_04.png'")

# =============================================================================
# STEP 9: SAVE RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 8: SAVING RESULTS")
print("-" * 70)

# Save results
nn_results = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
    'Score': [accuracy, precision, recall, f1, auc]
})
nn_results.to_csv('neural_network_results.csv', index=False)
print("✓ Results saved to 'neural_network_results.csv'")

# Save training history
history_df = pd.DataFrame(history)
history_df.to_csv('training_history.csv', index=False)
print("✓ Training history saved to 'training_history.csv'")

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': len(history['train_loss']),
    'best_val_auc': best_val_auc,
}, 'ivf_neural_network_final.pth')
print("✓ Model saved to 'ivf_neural_network_final.pth'")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 3 COMPLETE! 🎉")
print("=" * 70)
print(f"""
NEURAL NETWORK PROJECT COMPLETE!
{'='*70}

ARCHITECTURE:
  Input → 128 → 64 → 32 → Output
  Total parameters: {total_params:,}
  Dropout rate: 0.3
  Batch normalization: Yes

TRAINING:
  Epochs completed: {len(history['train_loss'])}
  Best validation AUC: {best_val_auc:.4f}
  Final learning rate: {history['learning_rate'][-1]:.6f}

FINAL TEST PERFORMANCE:
  • Accuracy:  {accuracy:.2%}
  • Precision: {precision:.4f}
  • Recall:    {recall:.4f}
  • F1-Score:  {f1:.4f}
  • AUC-ROC:   {auc:.4f}
""")

if baseline_auc is not None:
    print(f"""
COMPARISON TO BASELINE:
  Baseline AUC:     {baseline_auc:.4f}
  Neural Net AUC:   0.5425
  AUC after tweaks: {auc:.4f}
  Improvement:      {auc - baseline_auc:+.4f} ({((auc - baseline_auc) / baseline_auc * 100):+.2f}%)
  Improvement before tweak:      {auc - 0.5425} ({((auc - 0.5425) / 0.5425 * 100):+.2f}%)
""")

print(f"""
FILES CREATED:
  ✓ neural_network_results_kaggle.png
  ✓ neural_network_results.csv
  ✓ training_history.csv
  ✓ ivf_neural_network_final.pth

NEXT STEPS FOR IMPROVEMENT:
  1. Hyperparameter tuning (try different architectures)
  2. Feature engineering (create interaction terms)
  3. Ensemble methods (combine multiple models)
  4. Cross-validation for more robust evaluation
  5. Deploy as a web app (Streamlit/Flask)

{'='*70}
""")
