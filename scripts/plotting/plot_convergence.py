import matplotlib.pyplot as plt

iterations = list(range(1, 31))

recall = [0.124, 1.0056, 5.82917, 22.869, 44.1533, 55.2587, 59.6102, 61.322, 62.053, 62.3834, 62.5442, 62.6265, 62.6694, 62.6921, 62.7037, 62.7095, 62.7121, 62.7135, 62.7143, 62.7148, 62.7148, 62.715, 62.715, 62.7152, 62.7152, 62.7153, 62.7152, 62.7153, 62.7152, 62.7153]

updates = [2986792, 2969460, 2952825, 2924616, 2782702, 2149555, 1244372, 619178, 297689, 146468, 75159, 40827, 24418, 15194, 9744, 7315, 5944, 5403, 5118, 4918, 4803, 4753, 4729, 4719, 4702, 4702, 4686, 4690, 4688, 4689]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Standard NN-Descent Convergence Analysis', fontsize=16, fontweight='bold')

ax1.plot(iterations, recall, marker='o', color='b', linewidth=2)
ax1.axhline(y=80, color='r', linestyle='--', linewidth=2, label='SISAP 80% Threshold')
ax1.set_title('Graph Recall vs Iterations')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Recall (%)')
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend()

ax2.plot(iterations, updates, marker='s', color='orange', linewidth=2)
ax2.set_title('Graph Edge Updates vs Iterations')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Number of Updates')
ax2.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.savefig('plots/nndescent_convergence.png', dpi=300)
print("Plot saved as 'plots/nndescent_convergence.png'")