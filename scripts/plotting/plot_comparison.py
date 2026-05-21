import matplotlib.pyplot as plt

# --- FINAL DOCKER RESULTS (10,000 Queries, 8 Cores, Volume Mounts) ---
recalls = [100.0, 94.42, 85.57, 78.98, 60.08, 56.25]
search_times = [152.94, 53.38, 40.89, 20.15, 11.20, 5.86] 
labels = ['Baseline (FP32 1024D)', 'INT8 Quant (1024D)', 'PCA 512D', 'PCA 256D', 'PCA 128D', 'Binary Quant (1-bit)']
colors = ['gray', 'green', 'blue', 'orange', 'red', 'purple']
markers = ['o', '*', 's', 's', 's', 'X']

plt.figure(figsize=(11, 6))
plt.title('SISAP 2026 Preprocessing: Search Time vs. Recall (10k Queries)', fontsize=14, fontweight='bold')

# Plot each data point
for i in range(len(recalls)):
    plt.scatter(search_times[i], recalls[i], color=colors[i], marker=markers[i], s=200, label=labels[i], zorder=5)

# Add the 80% Win Condition Line
plt.axhline(y=80, color='red', linestyle='--', linewidth=2, label='SISAP 80% Threshold')
plt.fill_between([0, max(search_times)+10], 0, 80, color='red', alpha=0.1)

# Formatting
plt.xlabel('Search Execution Time (Seconds)', fontsize=12)
plt.ylabel('Max Possible Recall (%)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='center right', fontsize=10)

plt.xlim(-5, max(search_times) + 10) 

plt.tight_layout()
plt.savefig('plots/preprocessing_pareto_comparison.png', dpi=300)
print("Pareto comparison plot saved as 'plots/preprocessing_pareto_comparison.png'")