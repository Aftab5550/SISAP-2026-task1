import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import time

original_file = "data/benchmark-dev-wikipedia-bge-m3-small.h5"
dimensions_to_test = [512, 256, 128]

print("==================================================")
print("   SISAP 2026: MULTI-DIMENSIONAL PCA SWEEP        ")
print("==================================================")

try:
    with h5py.File(original_file, 'r') as f_in:
        print("[1] Loading original data into RAM (float32 for precise math)...")
        train_data = np.array(f_in['train'], dtype=np.float32)
        
        for target_dim in dimensions_to_test:
            print(f"\n--- Processing target dimension: {target_dim} ---")
            start_time = time.time()
            new_file = f"data/benchmark-dev-wikipedia-bge-m3-small-pca{target_dim}.h5"
            
            print("  -> Fitting PCA model...")
            pca = PCA(n_components=target_dim)
            train_reduced = pca.fit_transform(train_data)
            
            variance_kept = np.sum(pca.explained_variance_ratio_) * 100
            print(f"  -> Variance Retained: {variance_kept:.2f}%")
            
            # CRITICAL STEP: Re-normalize the vectors so Dot Product works correctly!
            print("  -> Applying L2-Normalization...")
            train_reduced_normalized = normalize(train_reduced, norm='l2')
            
            print(f"  -> Saving to {new_file}...")
            with h5py.File(new_file, 'w') as f_out:
                f_out.create_dataset('train', data=train_reduced_normalized.astype(np.float16))
                
                # Copy the official ground truth so our C++ baseline can score it
                allknn_group = f_out.create_group('allknn')
                allknn_group.create_dataset('knns', data=f_in['allknn/knns'])
                allknn_group.create_dataset('dists', data=f_in['allknn/dists'])
                
            print(f"  -> Done in {time.time() - start_time:.2f} seconds.")

    print("\n[SUCCESS] All PCA datasets generated successfully!")
    print("==================================================")

except FileNotFoundError:
    print(f"Error: Could not find '{original_file}'.")