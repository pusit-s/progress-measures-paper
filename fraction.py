import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import sys
import pandas as pd
import random
import os

# Import the provided Transformer module
sys.path.append('.')
try:
    from transformers import Transformer, Config
except ImportError:
    # Fallback if file not found locally (for demonstration)
    pass 

# ===========================
# 1. Configuration
# ===========================

# Experiment Setup
EXPERIMENT_STYLE = 'random'  # Options: 'random', 'strip', 'rect'
# If 'random': Uses FRACTIONS list. 'k' is ignored.
# If 'strip'/'rect': Uses 'k' parameter. FRACTIONS list is ignored (run manually or loop k).

FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
K_VALUE = 30  # Only used if style is 'strip' or 'rect'

# Model & Training Constants
P = 113
TOTAL_STEPS = 20000
LOG_EVERY = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1.0  

# Device Configuration (Explicitly disabling MPS as requested)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ===========================
# 2. Data Generation (Adjusted)
# ===========================

def create_data(p, fraction, style='random', k=30, seed=42):
    """
    Generates data for Modular Addition.
    
    Logic:
    - If style='random': Use 'fraction' to determine train size. Ignore 'k'.
    - If style='strip': Hold out rows 0..k for Test. Train on EVERYTHING else. Ignore 'fraction'.
    - If style='rect': Hold out box k*k for Test. Train on EVERYTHING else. Ignore 'fraction'.
    """
    # 1. Generate all pairs (a, b)
    all_pairs = [(i, j) for i in range(p) for j in range(p)]
    
    # 2. Define the Test Set based on Style
    if style == 'strip':
        # Logic from str.py: Test set is rows i in [0, k]
        # Train set is everything else (fraction is ignored)
        test_pairs_set = set((i, j) for i in range(k + 1) for j in range(p))
        
        train_pairs = [p for p in all_pairs if p not in test_pairs_set]
        test_pairs = list(test_pairs_set)
        
        print(f"Data Split (Strip k={k}): Train {len(train_pairs)}, Test {len(test_pairs)}")

    elif style == 'rect':
        # Logic from rec.py: Test set is square region [0, k] x [0, k]
        # Train set is everything else (fraction is ignored)
        test_pairs_set = set((i, j) for i in range(k + 1) for j in range(k + 1))
        
        train_pairs = [p for p in all_pairs if p not in test_pairs_set]
        test_pairs = list(test_pairs_set)
        
        print(f"Data Split (Rect k={k}): Train {len(train_pairs)}, Test {len(test_pairs)}")

    else: # style == 'random'
        # Logic: Random shuffle, then split by fraction
        import random
        random.seed(seed)
        random.shuffle(all_pairs)
        
        num_total = len(all_pairs)
        num_train = int(fraction * num_total)
        
        train_pairs = all_pairs[:num_train]
        test_pairs = all_pairs[num_train:]
        
        # print(f"Data Split (Random {fraction}): Train {len(train_pairs)}, Test {len(test_pairs)}")

    # 3. Convert to Tensors
    def to_tensors(pairs_list):
        if not pairs_list:
            return torch.empty((0, 2), dtype=torch.long).to(DEVICE), torch.empty((0), dtype=torch.long).to(DEVICE)
        
        data = torch.tensor(pairs_list, dtype=torch.long)
        x = data[:, 0]
        y = data[:, 1]
        labels = (x + y) % p
        inputs = torch.stack([x, y], dim=1)
        return inputs.to(DEVICE), labels.to(DEVICE)

    train_x, train_y = to_tensors(train_pairs)
    test_x, test_y = to_tensors(test_pairs)
    
    return train_x, train_y, test_x, test_y

# ===========================
# 3. Training & Evaluation
# ===========================

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

def run_training(var_value, run_style):
    """
    Runs one training session. 
    var_value is either 'fraction' (if random) or 'k' (if str/rec), 
    but for the 3x3 grid we assume iterating fractions.
    """
    
    # Determine arguments based on style
    if run_style == 'random':
        curr_frac = var_value
        curr_k = 0 # Ignored
        print(f"--- Training | Style: {run_style} | Fraction: {curr_frac} ---")
    else:
        curr_frac = 0 # Ignored
        curr_k = var_value # This would need to be passed if iterating K
        print(f"--- Training | Style: {run_style} | k: {curr_k} ---")

    # Generate Data
    train_x, train_y, test_x, test_y = create_data(
        P, 
        fraction=curr_frac, 
        style=run_style, 
        k=curr_k
    )
    
    # Model Config (Matches paper: 1 Layer, MLP enabled)
    cfg = Config(
        p=P,
        d_model=128,
        num_heads=4,
        d_mlp=512,       
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
    
    history_records = []
    
    for step in range(TOTAL_STEPS + 1):
        model.train()
        optimizer.zero_grad()
        
        logits = model(train_x)
        loss = criterion(logits[:, -1, :], train_y)
        loss.backward()
        optimizer.step()
        
        if step % LOG_EVERY == 0:
            train_loss, train_acc = evaluate(model, train_x, train_y, criterion)
            test_loss, test_acc = evaluate(model, test_x, test_y, criterion)
            
            history_records.append({
                'variable': var_value, # fraction or k
                'step': step,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'train_acc': train_acc,
                'test_acc': test_acc
            })
            
    return history_records

# ===========================
# 4. Main Execution
# ===========================

if __name__ == "__main__":
    all_data = []

    # We assume 'random' style for the 3x3 fraction grid request.
    # If you want to run 'rec' or 'strip', change EXPERIMENT_STYLE at the top
    # and provide a list of K values instead of FRACTIONS.
    
    loop_values = FRACTIONS if EXPERIMENT_STYLE == 'random' else [K_VALUE] 
    
    for val in tqdm(loop_values, desc="Runs"):
        history = run_training(val, EXPERIMENT_STYLE)
        all_data.extend(history)

    # Save CSV
    df = pd.DataFrame(all_data)
    csv_filename = "grokking_experiment_data.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nTraining complete. Data saved to {csv_filename}")

    # ===========================
    # 5. Plotting (3x3 Grid)
    # ===========================
    
    # Only generate 3x3 grid if we have 9 items (standard for the fraction sweep)
    if len(loop_values) == 9:
        
        # --- 1. Accuracy Plot ---
        fig_acc, axes_acc = plt.subplots(3, 3, figsize=(15, 12))
        axes_acc = axes_acc.flatten()
        
        for i, val in enumerate(loop_values):
            ax = axes_acc[i]
            subset = df[df['variable'] == val]
            
            ax.plot(subset['step'], subset['train_acc'], label='Train', color='blue')
            ax.plot(subset['step'], subset['test_acc'], label='Test', color='red')
            
            label_name = "Fraction" if EXPERIMENT_STYLE == 'random' else "k"
            ax.set_title(f"{label_name} {val}")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            
            if i >= 6: ax.set_xlabel("Steps")
            if i % 3 == 0: ax.set_ylabel("Accuracy")
            if i == 0: ax.legend(loc="lower right")

        plt.suptitle(f"Grokking Accuracy ({EXPERIMENT_STYLE})", fontsize=16)
        plt.tight_layout()
        plt.savefig("grokking_accuracy_grid.png", dpi=150)
        print("Saved accuracy plot.")

        # --- 2. Loss Plot ---
        fig_loss, axes_loss = plt.subplots(3, 3, figsize=(15, 12))
        axes_loss = axes_loss.flatten()
        
        for i, val in enumerate(loop_values):
            ax = axes_loss[i]
            subset = df[df['variable'] == val]
            
            ax.semilogy(subset['step'], subset['train_loss'], label='Train', color='blue')
            ax.semilogy(subset['step'], subset['test_loss'], label='Test', color='red')
            
            label_name = "Fraction" if EXPERIMENT_STYLE == 'random' else "k"
            ax.set_title(f"{label_name} {val}")
            ax.grid(True, which="both", ls="--", alpha=0.3)
            
            if i >= 6: ax.set_xlabel("Steps")
            if i % 3 == 0: ax.set_ylabel("Loss (Log Scale)")
            if i == 0: ax.legend(loc="upper right")

        plt.suptitle(f"Grokking Loss ({EXPERIMENT_STYLE})", fontsize=16)
        plt.tight_layout()
        plt.savefig("grokking_loss_grid.png", dpi=150)
        print("Saved loss plot.")
        
    else:
        print("Plotting skipped or basic plot generated (Run count != 9). check CSV for data.")
    
    plt.show()