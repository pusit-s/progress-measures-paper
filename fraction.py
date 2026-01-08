import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import sys
import pandas as pd
from transformers import Transformer, Config

# Import the provided Transformer module
sys.path.append('.')

# ===========================
# 1. Configuration
# ===========================

FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
P = 113
TOTAL_STEPS = 20000
LOG_EVERY = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1.0  

# Explicitly disable MPS (Apple Silicon) as requested, use CUDA or CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

torch.manual_seed(42)
np.random.seed(42)

# ===========================
# 2. Data & Model Setup
# ===========================

def create_data(p, fraction):
    """Generate modular addition data and split by fraction."""
    # Generate all pairs
    a = torch.arange(p)
    b = torch.arange(p)
    x, y = torch.meshgrid(a, b, indexing='ij')
    inputs = torch.stack([x.flatten(), y.flatten()], dim=1)
    targets = (inputs[:, 0] + inputs[:, 1]) % p
    
    # Shuffle
    perm = torch.randperm(inputs.shape[0])
    inputs = inputs[perm]
    targets = targets[perm]
    
    # Split
    num_train = int(fraction * inputs.shape[0])
    train_x, train_y = inputs[:num_train], targets[:num_train]
    test_x, test_y = inputs[num_train:], targets[num_train:]
    
    return train_x.to(DEVICE), train_y.to(DEVICE), test_x.to(DEVICE), test_y.to(DEVICE)

def evaluate(model, x, y, criterion):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        final_logits = logits[:, -1, :]
        loss = criterion(final_logits, y)
        preds = final_logits.argmax(dim=-1)
        acc = (preds == y).float().mean().item()
    model.train()
    return loss.item(), acc

def train_fraction(frac):
    print(f"--- Training Data Fraction: {frac} ---")
    train_x, train_y, test_x, test_y = create_data(P, frac)
    
    # Paper uses a 1-layer transformer with MLP
    cfg = Config(
        p=P,
        d_model=128,
        num_heads=4,
        d_mlp=512,       # Enabled MLP to match paper architecture
        num_layers=1,
        n_ctx=2,
        d_vocab=P+1,
        act_type='ReLU',
        use_ln=False,
        seed=42
    )
    
    model = Transformer(cfg).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    # Store history for this fraction
    history_records = []
    
    pbar = tqdm(range(TOTAL_STEPS), leave=False, mininterval=60)
    for step in pbar:
        model.train()
        optimizer.zero_grad()
        
        logits = model(train_x)
        loss = criterion(logits[:, -1, :], train_y)
        loss.backward()
        optimizer.step()
        
        # Log metrics
        if step % LOG_EVERY == 0:
            train_loss, train_acc = evaluate(model, train_x, train_y, criterion)
            test_loss, test_acc = evaluate(model, test_x, test_y, criterion)
            
            history_records.append({
                'fraction': frac,
                'step': step,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'train_acc': train_acc,
                'test_acc': test_acc
            })
            
    return history_records

# ===========================
# 3. Main Loop & CSV Saving
# ===========================

if __name__ == "__main__":
    all_data = []

    # Run experiment
    for frac in FRACTIONS:
        fraction_history = train_fraction(frac)
        all_data.extend(fraction_history)

    # Save to CSV
    df = pd.DataFrame(all_data)
    csv_filename = "grokking_experiment_data.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nTraining complete. Data saved to {csv_filename}")

    # ===========================
    # 4. Plotting
    # ===========================
    
    # --- Plot 1: Accuracy Grid (3x3) ---
    fig_acc, axes_acc = plt.subplots(3, 3, figsize=(15, 12))
    axes_acc = axes_acc.flatten()
    
    for i, frac in enumerate(FRACTIONS):
        ax = axes_acc[i]
        subset = df[df['fraction'] == frac]
        
        ax.plot(subset['step'], subset['train_acc'], label='Train', color='blue')
        ax.plot(subset['step'], subset['test_acc'], label='Test', color='red')
        
        ax.set_title(f"Data Fraction {frac}")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        
        if i >= 6: ax.set_xlabel("Steps")
        if i % 3 == 0: ax.set_ylabel("Accuracy")
        if i == 0: ax.legend(loc="lower right")

    plt.suptitle("Grokking: Train vs Test Accuracy", fontsize=16)
    plt.tight_layout()
    plt.savefig("grokking_accuracy_grid.png", dpi=150)
    print("Saved accuracy plot to grokking_accuracy_grid.png")

    # --- Plot 2: Loss Grid (3x3) ---
    fig_loss, axes_loss = plt.subplots(3, 3, figsize=(15, 12))
    axes_loss = axes_loss.flatten()
    
    for i, frac in enumerate(FRACTIONS):
        ax = axes_loss[i]
        subset = df[df['fraction'] == frac]
        
        # Log scale for loss (semilogy)
        ax.semilogy(subset['step'], subset['train_loss'], label='Train', color='blue')
        ax.semilogy(subset['step'], subset['test_loss'], label='Test', color='red')
        
        ax.set_title(f"Data Fraction {frac}")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        
        if i >= 6: ax.set_xlabel("Steps")
        if i % 3 == 0: ax.set_ylabel("Loss (Log Scale)")
        if i == 0: ax.legend(loc="upper right")

    plt.suptitle("Grokking: Train vs Test Loss", fontsize=16)
    plt.tight_layout()
    plt.savefig("grokking_loss_grid.png", dpi=150)
    print("Saved loss plot to grokking_loss_grid.png")
    
    plt.show()