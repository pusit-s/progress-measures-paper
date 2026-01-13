import torch
from fraction_config import FractionConfig
from fraction_trainer import FractionTrainer
import os

# Create a test config
config = FractionConfig(
    total_steps=20,
    log_every=5,
    collect_every=10,
    collect_activations=True,
    collect_layers=['blocks.0.mlp.hook_post', 'hook_embed']
)

# Initialize Trainer
trainer = FractionTrainer(config)

# Run a small training
print("Running test training...")
trainer.train_single_run(val=30, style='rect')

# Check if file exists
expected_file = "activations/rect_30_activations.pth"
if os.path.exists(expected_file):
    print(f"PASS: File {expected_file} exists.")
    data = torch.load(expected_file)
    print(f"Keys in data: {list(data.keys())}")
    
    # Check step 10
    if 10 in data:
        layer_keys = list(data[10].keys())
        print(f"Keys at step 10: {layer_keys}")
        if 'blocks.0.mlp.hook_post' in layer_keys and 'hook_embed' in layer_keys:
             print("PASS: Both layers found.")
        else:
             print("FAIL: Missing layers.")
             
        # Check shape
        act_shape = data[10]['blocks.0.mlp.hook_post'].shape
        print(f"Activation shape: {act_shape}")
        if act_shape == (113*113, 2, 512):
             print("PASS: Shape is correct.")
        else:
             print(f"FAIL: Incorrect shape {act_shape}.")
             
        # Check embed shape
        embed_shape = data[10]['hook_embed'].shape
        print(f"Embed shape: {embed_shape}")
        if embed_shape == (113*113, 2, 128):
            print("PASS: Embed shape is correct.")
        else:
            print(f"FAIL: Incorrect embed shape {embed_shape}.")

else:
    print(f"FAIL: File {expected_file} not found.")
