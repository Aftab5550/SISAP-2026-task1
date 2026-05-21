import matplotlib.pyplot as plt

iterations_15 = list(range(1, 31))
recall_15 = [0.124, 1.0056, 5.82917, 22.869, 44.1533, 55.2587, 59.6102, 61.322, 62.053, 62.3834, 62.5442, 62.6265, 62.6694, 62.6921, 62.7037, 62.7095, 62.7121, 62.7135, 62.7143, 62.7148, 62.7148, 62.715, 62.715, 62.7152, 62.7152, 62.7153, 62.7152, 62.7153, 62.7152, 62.7153]

iterations_32 = list(range(1, 10))
recall_32 = [0.586967, 12.7091, 50.8068, 69.6072, 74.5631, 76.1473, 76.7438, 76.9925, 77.1043]

iterations_40 = list(range(1, 11))
recall_40 = [0.9224, 21.547, 66.444, 80.1493, 83.2333, 84.1943, 84.5415, 84.6821, 84.7423, 84.7676]

iterations_48 = list(range(1, 10))
recall_48 = [1.33587, 31.6757, 76.8385, 85.9689, 87.8315, 88.3898, 88.584, 88.6583, 88.6901]

iterations_64 = list(range(1, 31))
recall_64 = [2.36213, 51.8922, 87.5723, 91.3278, 92.0022, 92.1929, 92.2551, 92.2742, 92.281, 92.2833, 92.2849, 92.2848, 92.2851, 92.285, 92.2851, 92.2847, 92.2853, 92.2849, 92.2852, 92.2849, 92.2852, 92.285, 92.2853, 92.2847, 92.2852, 92.2851, 92.2852, 92.2847, 92.2853, 92.285]

k_sizes = [15, 32, 40, 48, 64]
times = [64.5, 114.3, 199.4, 312.4, 1260.5]
final_recalls = [62.71, 77.10, 84.76, 88.69, 92.28]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Graph Oversampling Analysis: Recall Convergence and Execution Time', fontsize=16, fontweight='bold')

ax1.plot(iterations_15, recall_15, marker='o', color='gray', linewidth=2, label='K=15 (Standard)')
ax1.plot(iterations_32, recall_32, marker='^', color='orange', linewidth=2, label='K=32 (Under-Target)')
ax1.plot(iterations_40, recall_40, marker='v', color='blue', linewidth=2, label='K=40 (Min Viable)')
ax1.plot(iterations_48, recall_48, marker='s', color='green', linewidth=2, label='K=48 (Optimal)')
ax1.plot(iterations_64, recall_64, marker='D', color='purple', linewidth=2, label='K=64 (Slow)')
ax1.axhline(y=80, color='red', linestyle='--', linewidth=2, label='80% Threshold')
ax1.set_title('Recall Convergence per K-Size')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Recall (%)')
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend(loc='lower right')
ax1.set_xlim(0, 31)
ax1.set_ylim(0, 100)

ax2.plot(k_sizes, times, marker='o', color='firebrick', linewidth=2, markersize=8)
for i, txt in enumerate(final_recalls):
    ax2.annotate(f"{txt}%", (k_sizes[i], times[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, fontweight='bold')
ax2.set_title('Execution Time vs. K-Size')
ax2.set_xlabel('K-Build Size')
ax2.set_ylabel('Wall-Clock Time (Seconds)')
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.set_xticks(k_sizes)

plt.tight_layout()
plt.savefig('plots/master_oversampling_comparison.png', dpi=300)