
import torch
import sys
import os


files = [
    "saved_runs/5_digit_addition_finite.pth",
    "saved_runs/mod_addition_no_wd.pth",
    "saved_runs/wd_10-1_mod_addition_loss_curve.pth"
]

for path in files:
    print(f"--- Inspecting {path} ---")
    if not os.path.exists(path):
        print("File not found.")
        continue
    try:
        data = torch.load(path, map_location=torch.device('cpu'))
        print(f"Type: {type(data)}")
        if isinstance(data, dict):
            print(f"Keys: {data.keys()}")
            if 'config' in data:
                print(f"Config: {data['config']}")
            if 'model' in data:
                print("Model key found.")
        elif isinstance(data, list):
            print(f"List length: {len(data)}")
            if len(data) > 0:
                 print(f"First item type: {type(data[0])}")
    except Exception as e:
        print(f"Error loading: {e}")

