import h5py
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.abspath(os.path.join(current_dir, "..", "data", "dev", "benchmark-dev-wikipedia-bge-m3-small.h5"))

if len(sys.argv) > 1:
    filename = sys.argv[1]

print(f"--- Analyzing {filename} ---")

try:
    with h5py.File(filename, "r") as f:
        print("\nTop-level keys:")
        for key in f.keys():
            print(f" - {key}")
            
            if isinstance(f[key], h5py.Dataset):
                print(f"   Shape: {f[key].shape}")
                print(f"   Type:  {f[key].dtype}")
                print(f"   First vector preview: {f[key][0][:20]} ...")
            else:
                for subkey in f[key].keys():
                    print(f"   -> {subkey}")
                    if isinstance(f[key][subkey], h5py.Dataset):
                        print(f"      Shape: {f[key][subkey].shape}")
                        print(f"      Type:  {f[key][subkey].dtype}")
                        
except FileNotFoundError:
    print(f"Error: Could not find '{filename}'.")
    sys.exit(1)

print("\n--- Analysis Complete ---")