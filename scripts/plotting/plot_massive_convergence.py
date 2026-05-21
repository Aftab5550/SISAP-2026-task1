import matplotlib.pyplot as plt
import numpy as np
import os

iterations = np.arange(1, 9)
updates = [301920426, 267643303, 177597974, 101053696, 54583713, 30553695, 19361048, 14379494]
rates = [99.0553, 87.8095, 58.2671, 33.1541, 17.9080, 10.0242, 6.3520, 4.7176]

fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = '#1f77b4'
ax1.set_xlabel('Iteration', fontweight='bold')
ax1.set_ylabel('Total Edge Updates', color=color1, fontweight='bold')
ax1.plot(iterations, updates, marker='o', color=color1, linewidth=2.5, markersize=8)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, linestyle='--', alpha=0.7)

ax2 = ax1.twinx()

color2 = '#d62728'
ax2.set_ylabel('Update Rate (%)', color=color2, fontweight='bold')
ax2.plot(iterations, rates, marker='s', color=color2, linewidth=2.5, linestyle='dashed', markersize=8)
ax2.tick_params(axis='y', labelcolor=color2)

ax2.axhline(y=5.0, color='black', linestyle=':', linewidth=2, label='5% Early Stopping Threshold')
ax2.legend(loc='upper right', framealpha=0.9)

plt.title('NN-Descent Convergence on 6.35M Vectors (SISAP 2026)', fontsize=14, fontweight='bold', pad=15)
fig.tight_layout()

os.makedirs('plots/main', exist_ok=True)
plt.savefig('plots/main/massive_convergence.png', dpi=300, bbox_inches='tight')
print("Plot saved successfully to plots/main/massive_convergence.png")