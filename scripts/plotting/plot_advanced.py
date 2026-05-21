import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ====================================================================
# 1. DATA PREPARATION
# ====================================================================
print("Loading data...")
df = pd.read_csv('grid_results.csv')
df.columns = df.columns.str.strip()

# Convert time to minutes
df['COMPUTE_MINS'] = df['TOTAL_MATH'] / 60.0
df['CONFIG'] = 'K=' + df['K'].astype(str) + ', T=' + df['TREES'].astype(str) + ', R=' + df['MULT'].astype(str)

# Calculate Pareto Frontier
df_sorted = df.sort_values(by='COMPUTE_MINS')
pareto_front = []
max_recall = -1

for index, row in df_sorted.iterrows():
    if row['RECALL'] > max_recall:
        pareto_front.append(row)
        max_recall = row['RECALL']

pareto_df = pd.DataFrame(pareto_front)

# Find the critical milestones
max_recall_config = pareto_df.loc[pareto_df['RECALL'].idxmax()]
safe_configs = pareto_df[pareto_df['RECALL'] >= 85.0]
optimal_safe_config = safe_configs.sort_values('COMPUTE_MINS').iloc[0]

# Set global academic style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

# ====================================================================
# FIGURE 1: THE 4D MASTER PARETO FRONTIER
# ====================================================================
print("Generating 4D Pareto Frontier with Color-Matched Legend...")
fig1, ax1 = plt.subplots(figsize=(14, 8), dpi=300)

# Custom Palette to avoid Yellow (K=90 becomes Deep Orange)
custom_palette = {30: '#481567', 50: '#2d708e', 70: '#29af7f', 90: '#d55e00'}

# Scatter plot
sns.scatterplot(
    data=df, x='COMPUTE_MINS', y='RECALL', hue='K', size='TREES', style='MULT',
    sizes=(60, 260), palette=custom_palette, markers={6:'o', 7:'s', 8:'^', 9:'D'},
    alpha=0.75, edgecolor='black', linewidth=0.5, ax=ax1
)

# Draw Pareto Line
ax1.plot(
    pareto_df['COMPUTE_MINS'], pareto_df['RECALL'], 
    color='#e74c3c', linestyle='-', linewidth=2.5, zorder=2, label='Pareto Frontier'
)

# Draw Milestone Stars
ax1.scatter(
    optimal_safe_config['COMPUTE_MINS'], optimal_safe_config['RECALL'], 
    color='gold', marker='*', s=700, edgecolor='black', linewidth=1.5, zorder=4,
    label='Selected Milestones'
)
ax1.scatter(
    max_recall_config['COMPUTE_MINS'], max_recall_config['RECALL'], 
    color='gold', marker='*', s=700, edgecolor='black', linewidth=1.5, zorder=4
)

# Dynamically fetch the correct colors based on the K value of the milestone
opt_color = custom_palette[optimal_safe_config['K']]
max_color = custom_palette[max_recall_config['K']]

# Annotation 1: Optimal Target (Bottom Right - Top Box)
ax1.text(
    0.97, 0.05,  # x, y coordinates as a fraction of the axes (bottom right)
    f"★ Optimal Target (>85%)\n{optimal_safe_config['CONFIG']}\nTime: {optimal_safe_config['COMPUTE_MINS']:.1f}m | Recall: {optimal_safe_config['RECALL']:.1f}%",
    transform=ax1.transAxes,
    ha='right', va='bottom',
    fontsize=11.5, fontweight='bold', color='white',
    bbox=dict(boxstyle="round,pad=0.5", fc=opt_color, ec="black", lw=1.5, alpha=0.9),
    zorder=5
)

# Annotation 2: Absolute Ceiling (Bottom Right - Bottom Box)
ax1.text(
    0.97, 0.18,  # Just below the Optimal Target box
    f"★ Absolute Ceiling\n{max_recall_config['CONFIG']}\nTime: {max_recall_config['COMPUTE_MINS']:.1f}m | Recall: {max_recall_config['RECALL']:.1f}%",
    transform=ax1.transAxes,
    ha='right', va='bottom',
    fontsize=11.5, fontweight='bold', color='white',
    bbox=dict(boxstyle="round,pad=0.5", fc=max_color, ec="black", lw=1.5, alpha=0.9),
    zorder=5
)

# Add Threshold Lines
ax1.axhline(y=80.0, color='gray', linestyle='--', linewidth=1.5, label='Disqualification (80%)')
ax1.axhline(y=85.0, color='#27ae60', linestyle='--', linewidth=1.5, label='Target Threshold (85%)')

ax1.set_title("Apex Engine: Execution Time vs. Recall (Pareto Frontier)", fontsize=16, fontweight='bold', pad=15)
ax1.set_xlabel("Total CPU Compute Time (Minutes)", fontsize=13, fontweight='bold')
ax1.set_ylabel("True Recall (%)", fontsize=13, fontweight='bold')

# Move legend outside the plot
ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, shadow=True)

fig1.tight_layout()
fig1.savefig('1_pareto_master.png', format='png', bbox_inches='tight')

# ====================================================================
# FIGURE 2: HYPERPARAMETER ISOLATION (MARGINAL EFFECTS)
# ====================================================================
print("Generating Hyperparameter Isolation Dashboard...")
fig2, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300, sharey=True)
fig2.suptitle("Isolated Impact of Hyperparameters on Recall vs. Compute Time", fontsize=18, fontweight='bold', y=1.05)

params = [
    ('K', '#d55e00'),       # Deep Orange
    ('TREES', '#2d708e'),   # Steel Blue
    ('MULT', '#29af7f')     # Emerald Green
]
titles = ['Impact of Graph Width (K_BUILD)', 'Impact of Forest Breadth (NUM_TREES)', 'Impact of Deep Scoring (RERANK_MULT)']

for i, (param, color_hex) in enumerate(params):
    grouped = df.groupby(param)[['COMPUTE_MINS', 'RECALL']].mean().reset_index()
    
    sns.lineplot(
        data=grouped, x='COMPUTE_MINS', y='RECALL', marker='o', 
        markersize=11, linewidth=3.5, color=color_hex, ax=axes[i]
    )
    
    for index, row in grouped.iterrows():
        axes[i].text(
            row['COMPUTE_MINS'], row['RECALL'] + 0.15, 
            f"{param}={int(row[param])}", 
            horizontalalignment='center', fontsize=11, fontweight='bold', color='#333333'
        )

    axes[i].margins(x=0.15, y=0.15)
    
    axes[i].set_title(titles[i], fontsize=14, fontweight='bold')
    axes[i].set_xlabel("Average Compute Time (Minutes)", fontsize=12)
    if i == 0:
        axes[i].set_ylabel("Average True Recall (%)", fontsize=12)
    
    axes[i].grid(True, linestyle='--', alpha=0.7)

fig2.tight_layout()
fig2.savefig('2_hyperparameter_impact.png', format='png', bbox_inches='tight')

print("Success! Both formal plots generated.")