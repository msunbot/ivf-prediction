"""
IVF Success Prediction - Model Configurations
Defines all three model architectures and their training parameters
"""

import torch
import torch.nn as nn

# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

class IVFPredictorBalanced(nn.Module):
    """
    Balanced Model: Optimized for highest AUC
    
    Use Case:
    - Academic benchmarking
    - Research papers
    - Balanced precision/recall
    
    Performance:
    - AUC: 0.5425 (best)
    - Accuracy: 55%
    - Recall: 0.519
    """
    def __init__(self, input_size, dropout_rate=0.2):
        super(IVFPredictorBalanced, self).__init__()
        
        # Architecture: input → 32 → 16 → 1
        self.fc1 = nn.Linear(input_size, 32)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class IVFPredictorClinical(nn.Module):
    """
    Clinical Model: Optimized for highest recall
    
    Use Case:
    - Clinical decision support
    - Don't miss viable IVF candidates
    - False negatives more costly than false positives
    
    Performance:
    - AUC: 0.5323
    - Accuracy: 54.75%
    - Recall: 0.636 (best - catches 64% of successes!)
    """
    def __init__(self, input_size, dropout_rate=0.1):  # Lower dropout
        super(IVFPredictorClinical, self).__init__()
        
        # Architecture: input → 32 → 16 → 1 (same as balanced)
        self.fc1 = nn.Linear(input_size, 32)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class IVFPredictorProduction(nn.Module):
    """
    Production Model: Single layer for fast inference
    
    Use Case:
    - Mobile/edge deployment
    - Low-latency requirements
    - Resource-constrained environments
    - Need explainability
    
    Performance:
    - AUC: 0.5287
    - Accuracy: 50%
    - Recall: 0.481
    - Parameters: 937 (47% fewer than two-layer)
    - Overfitting: 0.0256 (best generalization)
    """
    def __init__(self, input_size, dropout_rate=0.2):
        super(IVFPredictorProduction, self).__init__()
        
        # Architecture: input → 24 → 1 (single hidden layer)
        self.fc1 = nn.Linear(input_size, 24)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(24, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# =============================================================================
# TRAINING CONFIGURATIONS
# =============================================================================

MODEL_CONFIGS = {
    "balanced": {
        "model_class": IVFPredictorBalanced,
        "name": "Balanced Model (Max AUC)",
        "dropout": 0.2,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "batch_size": 32,
        "max_epochs": 100,
        "patience": 15,
        "save_path": "models/best_model_balanced.pth",
        "description": "Optimized for highest AUC and balanced predictions",
        "metrics": {
            "auc": 0.5425,
            "accuracy": 0.55,
            "recall": 0.519,
            "precision": 0.569,
            "f1": 0.543
        }
    },
    
    "clinical": {
        "model_class": IVFPredictorClinical,
        "name": "Clinical Model (Max Recall)",
        "dropout": 0.1,  # Lower dropout for more learning
        "learning_rate": 0.0005,  # Slower, more careful learning
        "weight_decay": 1e-4,
        "batch_size": 32,
        "max_epochs": 100,
        "patience": 20,
        "save_path": "models/best_model_clinical.pth",
        "description": "Optimized to catch more IVF success cases (high recall)",
        "metrics": {
            "auc": 0.5323,
            "accuracy": 0.5475,
            "recall": 0.636,
            "precision": 0.553,
            "f1": 0.591
        }
    },
    
    "production": {
        "model_class": IVFPredictorProduction,
        "name": "Production Model (Fast & Stable)",
        "dropout": 0.2,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "batch_size": 32,
        "max_epochs": 50,  # Fewer epochs (converges faster)
        "patience": 15,
        "save_path": "models/best_model_production.pth",
        "description": "Single-layer model for fast inference and edge deployment",
        "metrics": {
            "auc": 0.5287,
            "accuracy": 0.50,
            "recall": 0.481,
            "precision": 0.516,
            "f1": 0.498
        }
    }
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_model(model_type, input_size):
    """
    Factory function to create model instances
    
    Args:
        model_type (str): One of ['balanced', 'clinical', 'production']
        input_size (int): Number of input features
    
    Returns:
        nn.Module: Initialized model
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Choose from {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_type]
    model_class = config["model_class"]
    dropout = config["dropout"]
    
    return model_class(input_size=input_size, dropout_rate=dropout)


def get_config(model_type):
    """Get full configuration for a model type"""
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}")
    return MODEL_CONFIGS[model_type]


def print_model_info():
    """Print comparison of all three models"""
    print("\n" + "=" * 80)
    print("IVF PREDICTION MODELS - COMPARISON")
    print("=" * 80)
    
    for model_type, config in MODEL_CONFIGS.items():
        print(f"\n📊 {config['name']}")
        print("-" * 80)
        print(f"Type: {model_type}")
        print(f"Description: {config['description']}")
        print(f"\nArchitecture:")
        if model_type == "production":
            print(f"  Input → 24 → Output (single hidden layer)")
        else:
            print(f"  Input → 32 → 16 → Output (two hidden layers)")
        print(f"  Dropout: {config['dropout']}")
        
        print(f"\nTraining Config:")
        print(f"  Learning Rate: {config['learning_rate']}")
        print(f"  Weight Decay: {config['weight_decay']}")
        print(f"  Batch Size: {config['batch_size']}")
        print(f"  Max Epochs: {config['max_epochs']}")
        print(f"  Patience: {config['patience']}")
        
        print(f"\nPerformance:")
        metrics = config['metrics']
        print(f"  AUC:       {metrics['auc']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    print("🎓 Research/Academic: Use 'balanced' (highest AUC)")
    print("🏥 Clinical Setting:  Use 'clinical' (highest recall)")
    print("⚡ Production/Mobile: Use 'production' (fastest, most stable)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Example usage
    print_model_info()
    
    # Create example models
    input_size = 37  # Number of features after preprocessing
    
    print("\nCreating model instances...")
    model_balanced = get_model("balanced", input_size)
    model_clinical = get_model("clinical", input_size)
    model_production = get_model("production", input_size)
    
    print(f"✓ Balanced model:   {sum(p.numel() for p in model_balanced.parameters()):,} parameters")
    print(f"✓ Clinical model:   {sum(p.numel() for p in model_clinical.parameters()):,} parameters")
    print(f"✓ Production model: {sum(p.numel() for p in model_production.parameters()):,} parameters")