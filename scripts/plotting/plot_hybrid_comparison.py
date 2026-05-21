import matplotlib.pyplot as plt

iterations = list(range(1, 31))

std_recall = [0.124, 1.0056, 5.82917, 22.869, 44.1533, 55.2587, 59.6102, 61.322, 62.053, 62.3834, 62.5442, 62.6265, 62.6694, 62.6921, 62.7037, 62.7095, 62.7121, 62.7135, 62.7143, 62.7148, 62.7148, 62.715, 62.715, 62.7152, 62.7152, 62.7153, 62.7152, 62.7153, 62.7152, 62.7153]
std_updates = [2986792, 2969460, 2952825, 2924616, 2782702, 2149555, 1244372, 619178, 297689, 146468, 75159, 40827, 24418, 15194, 9744, 7315, 5944, 5403, 5118, 4918, 4803, 4753, 4729, 4719, 4702, 4702, 4686, 4690, 4688, 4689]

hybrid_recall = [5.02697, 5.2461, 5.9559, 7.689, 11.4791, 18.708, 28.839, 38.8089, 46.1625, 50.8506, 53.6221, 55.2594, 56.2548, 56.8828, 57.2663, 57.5087, 57.6664, 57.7631, 57.8288, 57.8687, 57.8958, 57.9149, 57.9268, 57.9338, 57.9391, 57.9432, 57.9455, 57.9471, 57.9483, 57.949]
hybrid_updates = [497158, 466301, 1022384, 1684133, 2179495, 2453913, 2505027, 2301462, 1866035, 1363559, 920141, 593988, 381768, 243822, 156534, 102989, 68672, 44583, 31563, 21595, 15723, 12662, 9796, 7640, 6615, 6163, 5505, 5180, 4868, 4657]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Algorithmic Topology: Standard vs. PiPNN Hybrid Graph Construction', fontsize=16, fontweight='bold')

ax1.plot(iterations, std_recall, marker='o', color='b', linewidth=2, label='Standard NN-Descent')
ax1.plot(iterations, hybrid_recall, marker='s', color='g', linewidth=2, label='PiPNN Hybrid')
ax1.axhline(y=80, color='r', linestyle='--', linewidth=2, label='SISAP 80% Threshold')
ax1.set_title('Graph Recall Convergence', fontsize=13)
ax1.set_xlabel('Iteration', fontsize=11)
ax1.set_ylabel('Recall (%)', fontsize=11)
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend(loc='lower right', fontsize=10)

ax2.plot(iterations, std_updates, marker='o', color='b', linewidth=2, label='Standard Updates')
ax2.plot(iterations, hybrid_updates, marker='s', color='g', linewidth=2, label='Hybrid Updates')
ax2.set_title('Graph Edge Update Volume', fontsize=13)
ax2.set_xlabel('Iteration', fontsize=11)
ax2.set_ylabel('Number of Updates', fontsize=11)
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('plots/hybrid_vs_standard_comparison.png', dpi=300)
print("Plot saved as 'plots/hybrid_vs_standard_comparison.png'")