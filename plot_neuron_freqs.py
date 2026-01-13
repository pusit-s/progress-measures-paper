
import torch
import numpy as np
import matplotlib.pyplot as plt
import transformers
from transformers import Config, Transformer, calculate_key_freqs
import helpers
import os
import dataclasses
import einops

def main():
    # Load model
    path = "saved_runs/wd_10-1_mod_addition_loss_curve.pth"
    if not os.path.exists(path):
        print(f"Model file not found: {path}")
        return

    print(f"Loading model from {path}...")
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    
    # Reconstruct Config
    config_dict = checkpoint['config']
    # Filter out keys that are not in Config fields to avoid TypeError
    # (dataclass init doesn't like extra keys usually, unless __post_init__ handles it, 
    # but Config seems to just have fields.)
    # Let's assume standard Config fields.
    
    # We can use the dict unpacking if it matches exactly.
    # To be safe, let's create a default Config and update it?
    # Or just unpack. Config is a dataclass.
    try:
        config = Config(**config_dict)
    except TypeError as e:
        print(f"Config mismatch, attempting to filter keys: {e}")
        # Simplistic filtering if needed
        valid_keys = Config.__dataclass_fields__.keys()
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        config = Config(**filtered_dict)

    # FORCE CPU
    config = dataclasses.replace(config, device=torch.device('cpu'))

    # Initialize model
    model = Transformer(config)
    model.load_state_dict(checkpoint['model'])
    
    # Create all data
    print("Creating data...")
    all_data = torch.tensor([(i, j, config.p) for i in range(config.p) for j in range(config.p)]).to(config.device)

    # Calculate key frequencies
    print("Calculating key frequencies...")
    neuron_freqs, neuron_frac_explained = calculate_key_freqs(config, model, all_data, return_individual=True)

    # Plot histogram of key frequencies
    print("Plotting results...")
    plt.figure(figsize=(10, 6))
    
    # We want to plot the distribution of neuron_freqs
    # Bins: we have discrete frequencies from 1 to p//2
    bins = np.arange(1, config.p//2 + 2) - 0.5
    
    counts, _, _ = plt.hist(neuron_freqs, bins=bins, rwidth=0.8, color='skyblue', edgecolor='black')
    
    plt.title(f"Key Frequency of Activation of Neurons in Layer (p={config.p})")
    plt.xlabel("Frequency")
    plt.ylabel("Number of Neurons")
    plt.xticks(np.unique(neuron_freqs)) # Show checks only for present frequencies
    plt.grid(axis='y', alpha=0.5)
    
    output_image = "neuron_key_freq_distribution.png"
    plt.savefig(output_image)
    print(f"Saved plot to {output_image}")

    # Optional: Print detailed stats
    print("\nDetailed Statistics:")
    unique_freqs, counts = np.unique(neuron_freqs, return_counts=True)
    for freq, count in zip(unique_freqs, counts):
        print(f"Frequency {freq}: {count} neurons")

    # --- New: Plot Sin/Cos Components for Top Neurons ---
    print("\nPlotting Sin/Cos components for top neurons...")
    
    # 1. Identify top neurons (highest explained variance)
    # Sort indices by explained variance
    top_indices = np.argsort(neuron_frac_explained)[::-1]
    num_to_plot = 4
    top_neurons = top_indices[:num_to_plot]
    
    # 2. Prepare Fourier Basis
    fourier_basis = transformers.make_fourier_basis(config)
    
    # 3. Calculate components
    # We need the centered activations again
    # Recalculate or access from inside?
    # Since we can't easily access the inner variables of the previous function call without modification,
    # let's just re-calculate the activations for the chosen neurons. 
    # Or better, let's extract the Cos/Sin extraction logic into a loop here.
    
    # Rerun model to get activations (it's fast enough)
    # Or actually, we assume 'model' and 'all_data' are available.
    labels = torch.tensor([config.fn(i, j) for i, j, _ in all_data]).to(config.device)
    cache = {}
    model.remove_all_hooks()
    model.cache_all(cache)
    model(all_data)
    neuron_acts = cache['blocks.0.mlp.hook_post'][:, -1] # shape (p*p, d_mlp)
    neuron_acts_centered = neuron_acts - einops.reduce(neuron_acts, 'batch neuron -> 1 neuron', 'mean')
    
    # Create figure
    fig, axes = plt.subplots(num_to_plot, 1, figsize=(10, 4 * num_to_plot))
    if num_to_plot == 1: axes = [axes]
    
    for i, neuron_idx in enumerate(top_neurons):
        ax = axes[i]
        acts = neuron_acts_centered[:, neuron_idx] # shape (p*p,)
        
        cos_coeffs = []
        sin_coeffs = []
        freqs = range(1, config.p // 2 + 1)
        
        for k in freqs:
            # Calculate cos(k(x+y)) and sin(k(x+y)) coefficients
            # helpers.get_component_cos_xpy returns the PROJECTION (vector). 
            # We want the coefficient.
            # Since basis is normalized, coeff = dot(direction, vector) = direction @ vector
            
            # Note: helpers.get_component_cos_xpy(..., collapse_dim=True) returns the projection coefficient?
            # Let's verify line 191 in helpers.py: return (cos_xpy_direction @ tensor)
            # Yes, if tensor is vector, @ returns dot product (scalar).
            
            c_cos = helpers.get_component_cos_xpy(acts, k, fourier_basis, collapse_dim=True)
            c_sin = helpers.get_component_sin_xpy(acts, k, fourier_basis, collapse_dim=True)
            
            
            # Use Norm (magnitude) of the coefficient
            cos_coeffs.append(abs(c_cos.item()))
            sin_coeffs.append(abs(c_sin.item()))
            
        # Plot
        width = 0.4
        x = np.array(freqs)
        ax.bar(x - width/2, cos_coeffs, width=width, label='cos', color='blue', alpha=0.7)
        ax.bar(x + width/2, sin_coeffs, width=width, label='sin', color='red', alpha=0.7)
        
        key_freq = neuron_freqs[neuron_idx]
        frac = neuron_frac_explained[neuron_idx]
        ax.set_title(f"Neuron {neuron_idx} Spectrum (Key Freq: {key_freq}, Explained: {frac:.2f})")
        ax.set_ylabel("Norm of Fourier Component")
        ax.set_xlabel("Frequency k")
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Highlight key frequency
        # ax.axvline(key_freq, color='green', linestyle='--', alpha=0.5)

    plt.tight_layout()
    output_spect = "neuron_sin_cos_spectrum.png"
    plt.savefig(output_spect)
    print(f"Saved sin/cos spectrum plot to {output_spect}")


if __name__ == "__main__":
    main()
