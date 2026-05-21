import matplotlib.pyplot as plt

iterations = list(range(1, 31))

recall = [2.36213, 51.8922, 87.5723, 91.3278, 92.0022, 92.1929, 92.2551, 92.2742, 92.281, 92.2833, 92.2849, 92.2848, 92.2851, 92.285, 92.2851, 92.2847, 92.2853, 92.2849, 92.2852, 92.2849, 92.2852, 92.285, 92.2853, 92.2847, 92.2852, 92.2851, 92.2852, 92.2847, 92.2853, 92.285]

updates = [12797521, 12790838, 12375766, 7213154, 2927539, 1151750, 491729, 249469, 169025, 140512, 127763, 123492, 121039, 120321, 120358, 120194, 120281, 120228, 120295, 120101, 120349, 120137, 120282, 120165, 120326, 120137, 120312, 120140, 120300, 120177]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Oversampled NN-Descent: Convergence and List Thrashing Analysis', fontsize=16, fontweight='bold')

ax1.plot(iterations, recall, marker='o', color='purple', linewidth=2)
ax1.axhline(y=80, color='r', linestyle='--', linewidth=2, label='SISAP 80% Threshold')
ax1.axvline(x=5, color='gray', linestyle=':', linewidth=2, label='Optimal Stopping Point')
ax1.set_title('Recall Plateau (Compute Waste)', fontsize=13)
ax1.set_xlabel('Iteration', fontsize=11)
ax1.set_ylabel('Recall (%)', fontsize=11)
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend(loc='lower right')

ax2.plot(iterations, updates, marker='s', color='red', linewidth=2)
ax2.axvline(x=5, color='gray', linestyle=':', linewidth=2, label='Optimal Stopping Point')
ax2.set_title('Update Thrashing (INT8 Tie Swaps)', fontsize=13)
ax2.set_xlabel('Iteration', fontsize=11)
ax2.set_ylabel('Edge Updates', fontsize=11)
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('plots/oversampled_convergence.png', dpi=300)