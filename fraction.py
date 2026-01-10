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
import math

# Import the provided Transformer module
sys.path.append('.')
try:
    from transformers import Transformer, Config
except ImportError:
    pass # Expecting transformers.py to be present

# ===========================
# 1. Global Configuration
# ===========================

P = 113
TOTAL_STEPS = 20000
LOG_EVERY = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1.0  
SEED = 42

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ===========================
# 2. Data Generation Logic
# ===========================

def create_data(p, fraction, style=None, k=30, seed=42):
    """
    Generates data based on the specific style:
    - random (or None): Uses 'fraction' to split train/test.
    - strip: Holds out rows 0..k for Test. Ignores fraction.
    - rect: Holds out square 0..k x 0..k for Test. Ignores fraction.
    """
    random.seed(seed)
    all_pairs = [(i, j) for i in range(p) for j in range(p)]
    random.shuffle(all_pairs)
    
    if style == 'strip':
        # Hold out first k rows
        test_pairs_set = set((i, j) for i in range(k + 1) for j in range(p))
        train_pairs = [p for p in all_pairs if p not in test_pairs_set]
        test_pairs = [p for p in all_pairs if p in test_pairs_set]
        
    elif style == 'rect':
        # Hold out k x k square
        test_pairs_set = set((i, j) for i in range(k + 1) for j in range(k + 1))
        train_pairs = [p for p in all_pairs if p not in test_pairs_set]
        test_pairs = [p for p in all_pairs if p in test_pairs_set]
        
    else: 
        num_total = len(all_pairs)
        num_train = int(fraction * num_total)
        
        train_pairs = all_pairs[:num_train]
        test_pairs = all_pairs[num_train:]

    # Helper to convert to tensor
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
# 3. Training & Plotting Core
# ===========================

def evaluate(model, x, y, criterion):
    if len(x) == 0: return 0.0, 0.0
    model.eval()
    with torch.no_grad():
        logits = model(x)
        final_logits = logits[:, -1, :]
        loss = criterion(final_logits, y)
        preds = final_logits.argmax(dim=-1)
        acc = (preds == y).float().mean().item()
    model.train()
    return loss.item(), acc

def train_single_run(val, style):
    """
    Trains a single model.
    val: is 'fraction' if style='random', else it is 'k'.
    """
    # Determine args
    frac = val if style == 'random' else 0
    k_val = val if style != 'random' else 0
    
    # Generate Data
    train_x, train_y, test_x, test_y = create_data(P, fraction=frac, style=style, k=k_val, seed=SEED)
    
    # Model Config
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
        seed=SEED
    )
    
    model = Transformer(cfg).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    
    # Training Loop
    for step in range(TOTAL_STEPS + 1):
        model.train()
        optimizer.zero_grad()
        
        if len(train_x) > 0:
            logits = model(train_x)
            loss = criterion(logits[:, -1, :], train_y)
            loss.backward()
            optimizer.step()
        else:
            # Handle edge case if train set is empty (e.g. extreme k)
            loss = torch.tensor(0.0)
        
        if step % LOG_EVERY == 0:
            train_loss, train_acc = evaluate(model, train_x, train_y, criterion)
            test_loss, test_acc = evaluate(model, test_x, test_y, criterion)
            
            history.append({
                'variable': val,
                'step': step,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'train_acc': train_acc,
                'test_acc': test_acc
            })
            
    return history

def run_experiment_batch(name, style, values_list):
    print(f"\n=== Starting Experiment: {name} (Style: {style}) ===")
    print(f"Values to run: {values_list}")
    
    all_data = []
    
    # Run training for each value in the list
    for val in tqdm(values_list, desc=f"Running {name}"):
        run_history = train_single_run(val, style)
        all_data.extend(run_history)
        
    # Save CSV
    df = pd.DataFrame(all_data)
    csv_filename = f"{name}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Saved data to {csv_filename}")
    
    # Generate Plots
    print("Generating plots...")
    generate_plots(df, name, style, values_list)

def plot_dataset_distribution(ax, train_x, test_x, p):
    """
    Visualizes the dataset distribution (Train vs Test).
    Train_x and Test_x are (N, 2) tensors containing (i, j) pairs.
    Plotting logic:
    - x-axis: Column index (j)
    - y-axis: Row index (i)
    - Invert y-axis to match matrix visualization (row 0 at top).
    """
    # Convert to CPU numpy for plotting
    def to_cpu(t):
        return t.cpu().numpy()
        
    train_data = to_cpu(train_x)
    test_data = to_cpu(test_x)
    
    # In to_tensors: inputs = stack([x, y], dim=1) -> [i, j]
    # So col 0 is i (row), col 1 is j (col)
    
    # Plot Test first (background)
    if len(test_data) > 0:
        # Scatter(j, i) -> (col, row)
        ax.scatter(test_data[:, 1], test_data[:, 0], c='lightgrey', s=2, marker='s', label='Test')
        
    # Plot Train
    if len(train_data) > 0:
        ax.scatter(train_data[:, 1], train_data[:, 0], c='tab:blue', s=2, marker='s', label='Train')
        
    ax.set_xlim(-0.5, p - 0.5)
    ax.set_ylim(-0.5, p - 0.5)
    ax.invert_yaxis() # Put row 0 at the top
    ax.set_aspect('equal')
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

def generate_plots(df, name, style, values_list):
    """
    Generates a combined figure for the experiment batch.
    Rows: Each value in values_list
    Cols: 3 (Distribution, Loss, Accuracy)
    """
    num_vals = len(values_list)
    cols = 3
    rows = num_vals
    
    # Increase figure height to accommodate all rows
    fig_width = 15 # 5 * 3
    fig_height = 4 * rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    
    # Handle single row case where axes is 1D array
    if rows == 1:
        axes = np.array([axes])
        
    for i, val in enumerate(values_list):
        # --- Column 1: Dataset Distribution ---
        ax_dist = axes[i, 0]
        
        # Regenerate data to visualize the split
        frac = val if style == 'random' else 0
        k_val = val if style != 'random' else 0
        
        # Note: calling create_data again. Since seed is fixed, it should be deterministic.
        train_x, train_y, test_x, test_y = create_data(P, fraction=frac, style=style, k=k_val, seed=SEED)
        
        plot_dataset_distribution(ax_dist, train_x, test_x, P)
        
        label_title = f"{style.capitalize()} " + (f"Fraction {val}" if style == 'random' else f"k = {val}")
        ax_dist.set_title(label_title, fontsize=10, fontweight='bold')
        if i == 0: # Legend only on first row or specific place? 
            # Often cleaner without legend if colors are obvious, or put legend outside
            # Let's add a small legend
            ax_dist.legend(loc='upper right', fontsize=8)

        # Filter data for this value
        subset = df[df['variable'] == val]
        
        # --- Column 2: Loss ---
        ax_loss = axes[i, 1]
        
        if not subset.empty:
            ax_loss.semilogy(subset['step'], subset['train_loss'], label='Train', color='tab:blue')
            ax_loss.semilogy(subset['step'], subset['test_loss'], label='Test', color='lightgrey') # Grey for test match
            ax_loss.set_title("Loss", fontsize=10)
            ax_loss.grid(True, which="both", ls="--", alpha=0.3)
            if i == 0: ax_loss.legend()
        else:
            ax_loss.text(0.5, 0.5, "No Data", ha='center', va='center')

        # --- Column 3: Accuracy ---
        ax_acc = axes[i, 2]
        
        if not subset.empty:
            ax_acc.plot(subset['step'], subset['train_acc'], label='Train', color='tab:blue')
            ax_acc.plot(subset['step'], subset['test_acc'], label='Test', color='grey') # Grey for test match
            ax_acc.set_title("Accuracy", fontsize=10)
            ax_acc.set_ylim(-0.05, 1.05)
            ax_acc.grid(True, alpha=0.3)
            if i == 0: ax_acc.legend(loc='lower right')
        else:
            ax_acc.text(0.5, 0.5, "No Data", ha='center', va='center')

    plt.tight_layout()
    plot_filename = f"{name}_combined.png"
    plt.savefig(plot_filename, dpi=150)
    print(f"Saved combined plots to {plot_filename}")


# ===========================
# 4. Main Execution
# ===========================

if __name__ == "__main__":
    
    # 1. Fraction range 0.1 to 0.9 (random shuffle)
    # Using np.round to avoid floating point ugliness in filenames/titles
    fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    run_experiment_batch(
        name="experiment_1_fraction",
        style="random",
        values_list=fractions
    )
    
    # 2. Square range k=30 to 60 with step 5 (rect)
    # range(start, stop, step) -> stop is exclusive, so 65 to include 60
    k_square = [30, 35, 40, 45, 50, 55, 60] 
    run_experiment_batch(
        name="experiment_2_square",
        style="rect",
        values_list=k_square
    )
    
    # 3. Stripe range k=40 to 70 with step 10 (strip)
    # range(start, stop, step) -> stop is exclusive, so 80 to include 70
    k_stripe = [40, 50, 60, 70]
    run_experiment_batch(
        name="experiment_3_stripe",
        style="strip",
        values_list=k_stripe
    )
    
    print("\nAll experiments completed.")