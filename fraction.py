import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import sys
from transformers import Transformer, Config

# Add current directory to path so we can import the uploaded file
sys.path.append('.')

# ==========================================
# 1. Experiment Configuration
# ==========================================

# We will sweep fractions from 0.1 to 0.9
FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Constants for Modular Addition
P = 113
TOTAL_STEPS = 20000  # Steps to run for each fraction
LOG_EVERY = 200      # Log accuracy every N steps
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1.0   # High weight decay is crucial for grokking

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Set deterministic seeds
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 2. Helper Functions
# ==========================================

def create_data(p, fraction):
    """
    Generates modular addition data (a + b) % p and splits based on fraction.
    """
    # Create all pairs
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

def calculate_accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        # Prediction at the last token position
        preds = logits[:, -1, :].argmax(dim=-1)
        acc = (preds == y).float().mean().item()
    model.train()
    return acc

def run_training_for_fraction(fraction):
    """
    Trains a fresh model for a specific data fraction and returns the accuracy history.
    """
    print(f"\n--- Running for Fraction: {fraction} ---")
    
    # 1. Data
    train_x, train_y, test_x, test_y = create_data(P, fraction)
    
    # 2. Config & Model
    # We use the Config class from the uploaded file
    # We disable the MLP (d_mlp=0) to keep it an attention-only transformer 
    # similar to the simpler setups in grokking papers, but you can enable it if desired.
    cfg = Config(
        p=P,
        d_model=128,
        num_heads=4,
        d_mlp=0,        # Set to 0 for Attn-only, or 4*128 for standard
        num_layers=1,
        n_ctx=2,
        d_vocab=P+1,    # transformers.py usually expects p+1 for vocab
        act_type='ReLU',
        use_ln=False,
        seed=42
    )
    
    model = Transformer(cfg).to(DEVICE)
    
    # 3. Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()
    
    # 4. Loop
    steps_history = []
    test_acc_history = []
    grok_step = None
    
    pbar = tqdm(range(TOTAL_STEPS), leave=False)
    for step in pbar:
        model.train()
        optimizer.zero_grad()
        
        logits = model(train_x)
        final_logits = logits[:, -1, :]
        loss = criterion(final_logits, train_y)
        loss.backward()
        optimizer.step()
        
        if (step % LOG_EVERY == 0) or (step == TOTAL_STEPS - 1):
            test_acc = calculate_accuracy(model, test_x, test_y)
            steps_history.append(step)
            test_acc_history.append(test_acc)
            
            pbar.set_description(f"Frac {fraction} | Test Acc: {test_acc:.2f}")
            
            # Check if grokking occurred (defined as > 99% accuracy)
            if grok_step is None and test_acc > 0.99:
                grok_step = step
                
    return steps_history, test_acc_history, grok_step

# ==========================================
# 3. Main Execution Loop
# ==========================================

results = {}  # Store data for plotting

for frac in FRACTIONS:
    steps, accs, grok_step = run_training_for_fraction(frac)
    results[frac] = {
        'steps': steps,
        'accs': accs,
        'grok_step': grok_step
    }

print("\nAll training runs complete.")

# ==========================================
# 4. Plotting
# ==========================================

# Set up a colormap
colors = plt.cm.viridis(np.linspace(0, 1, len(FRACTIONS)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --- Plot 1: Test Accuracy Curves ---
for i, frac in enumerate(FRACTIONS):
    data = results[frac]
    ax1.plot(data['steps'], data['accs'], label=f'Frac {frac}', color=colors[i], linewidth=2)

ax1.set_xscale('log') # Grokking plots are usually semilog-x
ax1.set_xlabel('Optimization Steps', fontsize=12)
ax1.set_ylabel('Test Accuracy', fontsize=12)
ax1.set_title(f'Grokking Speed vs Data Fraction (p={P})', fontsize=14)
ax1.legend(title="Train Fraction")
ax1.grid(True, which="both", ls="--", alpha=0.5)

# --- Plot 2: Steps to Grok (Phase Transition) ---
# Filter out runs that never grokked (never reached 99%)
valid_fracs = [f for f in FRACTIONS if results[f]['grok_step'] is not None]
grok_steps = [results[f]['grok_step'] for f in valid_fracs]

# If some didn't grok, we can plot them at max_steps to indicate "infinite"
ax2.plot(valid_fracs, grok_steps, 'o-', color='black', linewidth=2, markersize=8)

ax2.set_xlabel('Training Data Fraction', fontsize=12)
ax2.set_ylabel('Steps to Reach 99% Accuracy', fontsize=12)
ax2.set_title('Data Efficiency "Phase Transition"', fontsize=14)
ax2.grid(True, ls="--", alpha=0.5)
# Invert y-axis? Often simpler to see "smaller is better". 
# But standard phase plots often show "Optimization steps required" dropping as Fraction increases.

plt.tight_layout()
plt.savefig('grokking_fraction_sweep.png', dpi=150)
print("Plot saved to 'grokking_fraction_sweep.png'")
plt.show()