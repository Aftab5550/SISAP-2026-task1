import matplotlib.pyplot as plt

iterations_base = list(range(1, 31))
recall_base = [0.124, 1.0056, 5.82917, 22.869, 44.1533, 55.2587, 59.6102, 61.322, 62.053, 62.3834, 62.5442, 62.6265, 62.6694, 62.6921, 62.7037, 62.7095, 62.7121, 62.7135, 62.7143, 62.7148, 62.7148, 62.715, 62.715, 62.7152, 62.7152, 62.7153, 62.7152, 62.7153, 62.7152, 62.7153]

iterations_int8 = list(range(1, 10))
recall_int8 = [1.33587, 31.6757, 76.8385, 85.9689, 87.8315, 88.3898, 88.584, 88.6583, 88.6901]

iterations_bit = list(range(1, 10))
recall_bit = [1.33607, 31.6317, 75.6079, 84.2935, 86.0323, 86.5459, 86.7267, 86.7943, 86.8233]

iterations_rpt = list(range(0, 7))
recall_rpt = [17.7292, 72.8348, 88.0562, 90.0835, 90.4964, 90.6091, 90.643]

iterations_apex = list(range(0, 7))
recall_apex = [17.8469, 72.097, 86.493, 88.3134, 88.678, 88.7716, 88.8002]

models = ['Standard', 'Pure INT8', 'Bitwise', 'RP-Tree', 'Apex Hardware']
times = [64.5, 312.4, 72.8, 51.5, 41.0]
final_recalls = [62.71, 88.69, 86.82, 90.64, 88.80]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('The Evolution of the Champion Engine', fontsize=16, fontweight='bold')

ax1.plot(iterations_base, recall_base, marker='o', color='gray', linewidth=2, label='Phase 1: Standard (64s)')
ax1.plot(iterations_int8, recall_int8, marker='s', color='red', linewidth=2, label='Phase 2: INT8 Oversampled (312s)')
ax1.plot(iterations_bit, recall_bit, marker='^', color='orange', linewidth=2, label='Phase 3: Bitwise Hybrid (73s)')
ax1.plot(iterations_rpt, recall_rpt, marker='D', color='green', linewidth=2, label='Phase 4: RP-Tree + Bitwise (51s)')
ax1.plot(iterations_apex, recall_apex, marker='*', color='blue', linewidth=3, label='Phase 5: Apex Hardware (41s)')

ax1.axhline(y=80, color='black', linestyle='--', linewidth=2, label='80% Threshold')
ax1.set_title('Convergence Trajectory')
ax1.set_xlabel('Iteration (0 = Initialization)')
ax1.set_ylabel('Recall (%)')
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend(loc='lower right')
ax1.set_xlim(0, 31)
ax1.set_ylim(0, 100)

colors = ['gray', 'red', 'orange', 'green', 'blue']
bars = ax2.bar(models, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_title('Execution Time vs. Final Recall')
ax2.set_ylabel('Total Time (Seconds)')
ax2.grid(axis='y', linestyle=':', alpha=0.7)

for bar, rec in zip(bars, final_recalls):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 5, f'{height}s\n({rec}%)', ha='center', va='bottom', fontweight='bold')

ax2.set_ylim(0, 360)

plt.tight_layout()
plt.savefig('plots/final_architecture_evolution.png', dpi=300)