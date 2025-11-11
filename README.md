# 🧬 IVF Success Prediction using Machine Learning

> **Predicting IVF live birth outcomes using patient characteristics and neural networks**
A complete machine learning pipeline for predicting IVF (In Vitro Fertilization) success rates, featuring data exploration, preprocessing, baseline modeling, and three specialized neural network architectures optimized for different use cases.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Model Zoo](#model-zoo)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Results](#results)
- [Lessons Learned](#lessons-learned)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

This project tackles the challenging problem of predicting IVF live birth success using machine learning. IVF outcomes depend on numerous interacting factors including:

- **Patient demographics**: Age, BMI, years of infertility
- **Hormonal markers**: AMH, FSH, LH, Estrogen, Progesterone
- **Uterine factors**: Endometrial thickness, AFC count
- **Treatment details**: Protocol type, embryo quality, transfer type
- **Lifestyle factors**: Diet, exercise, stress levels, sleep
- **Male factors**: Sperm parameters (count, motility, morphology)

The project demonstrates a complete ML workflow from data exploration to model deployment, with emphasis on understanding trade-offs between different modeling approaches.

---

## ✨ Key Features

- **🔍 Comprehensive Data Exploration**: Detailed analysis of 2,000 patient records with 40+ features
- **🧹 Domain-Aware Preprocessing**: Intelligent handling of missing values based on medical context
- **📊 Multiple Model Architectures**: Three specialized models optimized for different priorities
- **⚖️ Trade-off Analysis**: Demonstrates when to prioritize AUC vs. recall vs. generalization
- **📈 Extensive Visualization**: Training curves, confusion matrices, ROC curves, prediction distributions
- **🏥 Clinical Focus**: Considers real-world implications of false positives vs. false negatives
- **📝 Well-Documented**: Extensive comments, learning log, and analysis of experiments

---

## 📊 Dataset

**Source**: Kaggle IVF Success Prediction Dataset  
**Size**: 2,000 patients  
**Features**: 40+ clinical, lifestyle, and treatment variables  
**Target**: Live birth success (binary: 0/1)  
**Class Distribution**: 51.4% success, 48.6% failure (balanced)

### Key Features Used:
```
Numerical (24):
├── Patient Demographics: Age, BMI, Years_of_Infertility
├── Hormonal Markers: AMH_Level, FSH_Level, LH_Level, Estrogen_E2, Progesterone_P4
├── Uterine Factors: Endometrial_Thickness_mm, AFC_Count
├── Metabolic: Thyroid_TSH, Insulin_Level
├── Treatment: Number_of_IVF_Cycles, Pregnancy_History, Number_of_Embryos_Transferred
├── Lifestyle: Diet_Quality_Score, Yoga_Sessions, Stress_Level, Physical_Activity, Sleep_Duration
└── Male Factors: Sperm_Count, Sperm_Motility, Sperm_Morphology, Sperm_DNA_Fragmentation

Categorical (13):
├── Medical: Diagnosis_Type, Male_Infertility_Diagnosis
├── Treatment: Medication_Protocol, Transfer_Type, ICSI_or_IVF, Day_of_Transfer
├── Procedures: Luteal_Support_Given, PGT, Assisted_Hatching_Used
└── Lifestyle: Smoking_Status, Alcohol_Consumption, Exposure_to_Environmental_Toxins, Occupation_Type
```

---

## 🏆 Model Zoo

This project includes **three specialized models**, each optimized for different priorities:

| Model | Architecture | Use Case | AUC | Recall | Parameters | Speed |
|-------|-------------|----------|-----|--------|------------|-------|
| **Balanced** | 37→32→16→1 | Academic benchmark | **0.5425** | 0.519 | 1,761 | Medium |
| **Clinical** | 37→32→16→1 | Max viable cases | 0.5323 | **0.636** | 1,761 | Medium |
| **Production** | 37→24→1 | Edge deployment | 0.5287 | 0.481 | **937** | **Fast** |

### 1️⃣ Balanced Model (Recommended for Research)
```python
Use when: Maximizing AUC, academic benchmarking
✓ Highest AUC (0.5425, +4.2% vs baseline)
✓ Best overall performance (55% accuracy)
✓ Balanced precision/recall
⚠ More parameters (1,761)
```

### 2️⃣ Clinical Model (Recommended for Healthcare)
```python
Use when: Clinical decision support, don't miss viable cases
✓ Highest recall (0.636 - catches 64% of successes!)
✓ 22% better at finding viable IVF candidates than balanced model
✓ Lower false negative rate (important for patient decisions)
⚠ Slightly lower AUC (0.5323)
```

### 3️⃣ Production Model (Recommended for Deployment)
```python
Use when: Mobile apps, edge devices, resource constraints
✓ Fastest inference (47% fewer parameters)
✓ Best generalization (overfitting gap: 0.0256)
✓ Most stable training
⚠ Lower accuracy (50%)
```

---

## 🔧 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ivf-prediction.git
cd ivf-prediction
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Requirements
```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
torch>=2.0.0
joblib>=1.0.0
```

---

## 🚀 Quick Start

### Option 1: Run Complete Pipeline
```bash
# Step 1: Data exploration
python 01_data_exploration.py

# Step 2: Baseline model (Logistic Regression)
python 02_preprocessing_baseline.py

# Step 3: Neural network (choose model type inside script)
python 03_neural_network.py
```

### Option 2: Train Specific Model
```python
from models.model_configs import get_model, get_config

# Choose model type: 'balanced', 'clinical', or 'production'
model = get_model('clinical', input_size=37)
config = get_config('clinical')

# Train with config parameters
# See 03_neural_network.py for full training code
```

### Option 3: Make Predictions
```python
import torch
from models.model_configs import get_model

# Load trained model
model = get_model('balanced', input_size=37)
model.load_state_dict(torch.load('models/best_model_balanced.pth'))
model.eval()

# Prepare your data (must match training preprocessing)
# X_new = preprocess(your_data)

# Make predictions
with torch.no_grad():
    probabilities = model(X_new)
    predictions = (probabilities >= 0.5).float()
```

---

## 📁 Project Structure

```
ivf-prediction/
├── data/
│   └── kaggle_data_set.csv              # Raw dataset
│
├── models/
│   ├── model_configs.py                 # Model definitions & configs
│   ├── best_model_balanced.pth          # Trained: Balanced model
│   ├── best_model_clinical.pth          # Trained: Clinical model
│   ├── best_model_production.pth        # Trained: Production model
│   ├── logistic_regression_model.pkl    # Baseline model
│   ├── scaler.pkl                       # Feature scaler
│   └── label_encoders.pkl               # Categorical encoders
│
├── results/
│   ├── ivf_kaggle_exploration.png       # EDA visualizations
│   ├── baseline_model_evaluation_kaggle.png
│   └── neural_network_results_kaggle.png
│
├── 01_exploration.py                    # Data exploration
├── 02_preprocessing_baseline.py         # Baseline (Logistic Regression)
├── 03_neural_network.py                 # Neural network training
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
├── LEARNING_LOG.md                      # Detailed learning journey
└── .gitignore                           # Git ignore rules
```

---

## 📈 Results

### Model Performance Comparison

| Metric | Baseline (LR) | Balanced NN | Clinical NN | Production NN |
|--------|--------------|-------------|-------------|---------------|
| **AUC** | 0.5203 | **0.5425** (+4.2%) | 0.5323 (+2.3%) | 0.5287 (+1.6%) |
| **Accuracy** | 52.3% | **55.0%** | 54.75% | 50.0% |
| **Precision** | 0.534 | **0.569** | 0.553 | 0.516 |
| **Recall** | 0.512 | 0.519 | **0.636** | 0.481 |
| **F1-Score** | 0.523 | 0.543 | **0.591** | 0.498 |
| **Overfitting Gap** | N/A | 0.0766 | 0.0422 | **0.0256** |
| **Parameters** | N/A | 1,761 | 1,761 | **937** |

### Key Findings

1. **All neural networks beat the baseline** by 1.6-4.2% in AUC
2. **Clinical model catches 22% more successes** than balanced model (recall: 0.636 vs 0.519)
3. **Production model has best generalization** (lowest overfitting)
4. **Top 5 predictive features** (by correlation with outcome):
   - Estrogen_E2_Level (+)
   - Age (-)
   - AMH_Level (+)
   - Number_of_Embryos_Transferred (+)
   - Endometrial_Thickness_mm (+)

### Trade-off Analysis

**Choosing Clinical over Balanced:**
- ✅ +11.7% improvement in recall (0.519 → 0.636)
- ✅ +8.8% improvement in F1-score
- ✅ 45% reduction in overfitting
- ⚠️ -1.8% decrease in AUC

**For IVF prediction, this trade-off is worthwhile** because:
- Missing a viable case (false negative) = patient may give up
- False positive = patient tries again (no harm, just emotional impact)

---

## 💡 Lessons Learned

### Technical Insights

1. **Simpler is Often Better**
   - Started with 128→64→32 architecture (15K parameters)
   - Failed spectacularly (AUC 0.504, severe overfitting)
   - Simplified to 32→16 (1.7K parameters)
   - **Result**: +8.2% AUC improvement, 45% less overfitting

2. **Class Imbalance Handling**
   - Dataset is nearly balanced (51/49 split)
   - Adding `pos_weight` to loss function made predictions worse
   - **Lesson**: Don't blindly apply techniques; check if problem exists first

3. **Missing Data Context Matters**
   - 689 missing values for "Male_Infertility_Diagnosis"
   - Naive approach: Fill with most common value ("Varicocele")
   - **Problem**: Labels 689 healthy men as having a condition!
   - **Solution**: Fill with "None" (missing = no diagnosis)
   - **Impact**: Model learned truthful patterns

4. **Multiple Metrics Matter**
   - AUC isn't everything
   - For clinical applications, recall can be more important
   - Always consider the **cost of different error types**

5. **Overfitting Detection**
   - Train/val gap > 0.10 = severe overfitting
   - Solutions: Reduce model size, increase dropout, add regularization
   - Early stopping is essential

### ML Best Practices Demonstrated

✅ Always split data BEFORE scaling (prevent data leakage)  
✅ Use stratified splitting for balanced train/test sets  
✅ Implement early stopping to prevent overfitting  
✅ Start with simple baseline (logistic regression)  
✅ Monitor both training AND validation metrics  
✅ Consider domain knowledge when preprocessing  
✅ Document experiments (successful AND failed)  
✅ Create multiple models for different use cases  

---

## 🚧 Future Work

### Short-term Improvements
- [ ] **Cross-validation**: 5-fold CV for more robust evaluation
- [ ] **Hyperparameter optimization**: Grid search or Optuna
- [ ] **Feature engineering**: Create interaction terms (Age × AMH, BMI × FSH)
- [ ] **Ensemble methods**: Combine multiple models (stacking)
- [ ] **Explainability**: SHAP values for feature importance

### Long-term Extensions
- [ ] **Multi-task learning**: Predict clinical pregnancy, miscarriage, twins
- [ ] **Survival analysis**: Time-to-pregnancy modeling
- [ ] **Web interface**: Streamlit or Flask app for predictions
- [ ] **Transfer learning**: Fine-tune on clinic-specific data
- [ ] **Mobile deployment**: TensorFlow Lite for on-device inference
- [ ] **Fairness analysis**: Check for demographic biases

---

## 🤝 Contributing

Contributions are welcome! This project was built as a learning exercise, and suggestions for improvement are appreciated.

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

### Areas for Contribution
- Additional model architectures
- Better visualization techniques
- Deployment examples (Docker, cloud services)
- Unit tests for preprocessing functions
- Documentation improvements

---

## 📝 License

This project is licensed under the MIT License

---

## 🙏 Acknowledgments

- **Kaggle** for providing the IVF prediction dataset
- **Deep Learning Specialization (Coursera)** for foundational ML knowledge
- **PyTorch** and **scikit-learn** communities for excellent documentation
- **IVF research community** for domain expertise

---

## 📧 Contact

**Author**: Michelle Sun  
**GitHub**: [@msunbot](https://github.com/msunbot)  

---

## 🌟 Star This Project

If you found this project helpful or interesting, please consider giving it a star! ⭐

It helps others discover the project and motivates continued development.

---

**Built with ❤️ as a learning project in machine learning and healthcare AI**
