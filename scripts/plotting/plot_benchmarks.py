import matplotlib.pyplot as plt
import numpy as np

datasets = ['CCNEWS\n(Clustered)', 'YAHOOAQ\n(Sparse)', 'GOOAQ\n(Large Sparse)']
vectors = np.array([0.603664, 0.670164, 3.001496])
times = np.array([58.45, 79.46, 294.21])
recalls = np.array([96.539, 99.881, 99.751])

iters = np.arange(1, 6)
ccnews_updates = [80.56, 40.06, 14.59, 5.08, 1.82]
yahooaq_updates = [98.08, 63.39, 23.00, 7.68, 2.49]
gooaq_updates = [69.79, 22.99, 7.67, 2.73]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

bars = ax1.bar(datasets, recalls, color=['#2c3e50', '#2980b9', '#3498db'], width=0.5)
ax1.set_ylim(90, 101)
ax1.set_ylabel('Recall@15 (%)', fontweight='bold')
ax1.set_title('Algorithm Robustness Across Distributions', fontweight='bold', pad=15)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

for bar in bars:
    height = bar.get_height()
    ax1.annotate(f'{height:.2f}%',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 4),
                 textcoords="offset points",
                 ha='center', va='bottom', fontweight='bold')

ax2.plot(vectors, times, marker='o', markersize=8, linestyle='-', color='#e74c3c', linewidth=2.5)
ax2.set_xlim(0, 3.5)
ax2.set_ylim(0, 350)
ax2.set_xlabel('Dataset Size (Millions of Vectors)', fontweight='bold')
ax2.set_ylabel('Execution Time (Seconds)', fontweight='bold')
ax2.set_title('Engine Scaling\n(Execution Time vs. Volume)', fontweight='bold', pad=15)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax2.annotate(f'CCNEWS\n{times[0]}s', (vectors[0], times[0]), textcoords="offset points", xytext=(12, -15), ha='left', fontweight='bold', color='#2c3e50')
ax2.annotate(f'YAHOOAQ\n{times[1]}s', (vectors[1], times[1]), textcoords="offset points", xytext=(-12, 10), ha='right', fontweight='bold', color='#2c3e50')
ax2.annotate(f'GOOAQ\n{times[2]}s', (vectors[2], times[2]), textcoords="offset points", xytext=(-12, 10), ha='right', fontweight='bold', color='#2c3e50')

ax3.plot(iters, ccnews_updates, marker='s', markersize=6, linestyle='-', color='#2c3e50', linewidth=2, label='CCNEWS')
ax3.plot(iters, yahooaq_updates, marker='^', markersize=6, linestyle='-', color='#2980b9', linewidth=2, label='YAHOOAQ')
ax3.plot(iters[:4], gooaq_updates, marker='o', markersize=6, linestyle='-', color='#3498db', linewidth=2, label='GOOAQ')

ax3.set_xlabel('NN-Descent Iteration', fontweight='bold')
ax3.set_ylabel('Graph Update Rate (%)', fontweight='bold')
ax3.set_title('Convergence Curve\n(Graph Stability over Iterations)', fontweight='bold', pad=15)
ax3.set_xticks(iters)
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.legend(frameon=False)

plt.tight_layout()
plt.savefig('benchmark_analysis.png', dpi=300, bbox_inches='tight')