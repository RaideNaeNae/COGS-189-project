import os
from pathlib import Path

# 1. SET THIS TO THE NAME OF THE FAILING PARTICIPANT
fail_name = "angelo" # or whoever is failing

script_dir = Path(__file__).parent.resolve()
data_dir = script_dir.parent / "data"

# Find the folder
match = list(data_dir.glob(f"*{fail_name}*"))

if match:
    target = match[0]
    print(f"--- DIAGNOSTICS FOR: {fail_name.upper()} ---")
    print(f"Found folder: {target}")
    
    # List EVERY single file inside that folder to find the hidden .npy
    print("\nListing all files found inside:")
    count = 0
    for root, dirs, files in os.walk(target):
        for file in files:
            print(f"  -> Found: {file}")
            count += 1
    
    if count == 0:
        print("❌ This folder is actually empty!")
else:
    print(f"❌ Could not even find a folder for '{fail_name}' in {data_dir}")