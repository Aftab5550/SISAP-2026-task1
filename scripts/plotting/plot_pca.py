import matplotlib.pyplot as plt

# ==========================================
# FINAL DOCKER EXPERIMENTAL DATA (10,000 Queries, 8 Cores, Volume Mounts)
# ==========================================
dimensions = [1024, 512, 256, 128]

# Preprocessing Metrics (From Python)
variance_retained = [100.0, 97.62, 84.64, 64.34]
pca_fit_time = [0, 677.51, 270.61, 133.50] 

# Search Metrics (From your final Docker logs)
recall_percent = [100.0, 85.57, 78.98, 60.08]
algo_time_sec = [152.94, 40.89, 20.15, 11.20]
# Converted MB to GB accurately (e.g., 941 MB / 1024 = 0.92 GB)
peak_ram_gb = [1.38, 1.00, 0.92, 0.77] 

# ==========================================
# PLOTTING SETUP
# ==========================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('SISAP 2026: Complete PCA Pipeline Analysis (10k Queries)', fontsize=18, fontweight='bold')

def format_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)

# --- PANEL 1: Information Loss ---
ax1 = axes[0, 0]
ax1.plot(dimensions, variance_retained, marker='o', color='purple', linewidth=2, markersize=8)
format_ax(ax1, 'Information Loss (PCA)', 'Dimensions', 'Variance Retained (%)')
ax1.invert_xaxis() 

# --- PANEL 2: Max Possible Recall ---
ax2 = axes[0, 1]
ax2.plot(dimensions, recall_percent, marker='o', color='blue', linewidth=2, markersize=8)
format_ax(ax2, 'Absolute Recall Ceiling', 'Dimensions', 'Max Possible Recall (%)')
ax2.invert_xaxis()
ax2.axhline(y=80, color='red', linestyle='-', linewidth=2, label='SISAP 80% Win Condition')
ax2.fill_between(dimensions, 0, 80, color='red', alpha=0.1) 
ax2.legend(loc='lower left')

# --- PANEL 3: PCA Fitting Time ---
ax3 = axes[0, 2]
ax3.bar([str(d) for d in dimensions], pca_fit_time, color='orange', alpha=0.8, edgecolor='black')
format_ax(ax3, 'PCA Fitting Time (Cost)', 'Dimensions', 'Time (Seconds)')
ax3.grid(axis='y', linestyle='--', alpha=0.7) 

# --- PANEL 4: Algorithm Execution Time ---
ax4 = axes[1, 0]
ax4.plot(dimensions, algo_time_sec, marker='s', color='darkorange', linewidth=2, markersize=8)
format_ax(ax4, 'Search Execution Time', 'Dimensions', 'Time (Seconds)')
ax4.invert_xaxis()

# --- PANEL 5: Measured Peak RAM ---
ax5 = axes[1, 1]
ax5.plot(dimensions, peak_ram_gb, marker='^', color='green', linewidth=2, markersize=8)
format_ax(ax5, 'Measured Peak RAM Usage', 'Dimensions', 'Memory Used (GB)')
ax5.set_ylim(0.5, 1.5)
ax5.invert_xaxis()

axes[1, 2].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('plots/complete_pca_analysis.png', dpi=300)
print("Updated plot saved as 'plots/complete_pca_analysis.png'!")