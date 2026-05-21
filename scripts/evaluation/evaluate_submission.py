import h5py
import numpy as np
import sys

def evaluate(gt_file, pred_file):
    with h5py.File(gt_file, 'r') as f_gt:
        gt_data = f_gt['/allknn/knns'][:]
    
    with h5py.File(pred_file, 'r') as f_pred:
        pred_data = f_pred['knns'][:]
        build_time = f_pred.attrs.get('buildtime', 0.0)
        algo_raw = f_pred.attrs.get('algo', 'Unknown')
        algo_name = algo_raw.decode('utf-8') if isinstance(algo_raw, bytes) else str(algo_raw)
    
    k_eval = pred_data.shape[1]
    num_queries = pred_data.shape[0]
    
    hits = 0
    for i in range(num_queries):
        pred_row = pred_data[i]
        gt_row = gt_data[i]
        
        gt_filtered = gt_row[gt_row != (i + 1)][:k_eval]
        hits += len(np.intersect1d(pred_row, gt_filtered))
        
    recall = hits / (num_queries * k_eval)
    
    print("==================================================")
    print("   SISAP 2026: INDEPENDENT EVALUATION FRAMEWORK   ")
    print("==================================================")
    print(f"Algorithm:   {algo_name}")
    print(f"Vectors:     {num_queries}")
    print(f"Build Time:  {build_time} seconds")
    print(f"True Recall: {recall * 100:.4f}%")
    print("==================================================")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(1)
    evaluate(sys.argv[1], sys.argv[2])