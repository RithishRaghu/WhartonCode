import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Load your actual CSV data
# ==============================

# Example: "whl_2025.csv" (replace with your file path)
csv_file = "your_file.csv"

# Read CSV
data = pd.read_csv(csv_file)

# Ensure columns exist
required_columns = ["Team", "PowerRank", "Disparity"]
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"CSV must have a '{col}' column.")

# Sort by PowerRank (1 = strongest at top)
data = data.sort_values("PowerRank").reset_index(drop=True)

# Sort data by PowerRank (1 = strongest at the top)
data = data.sort_values("PowerRank").reset_index(drop=True)

# Correlation
correlation = np.corrcoef(data["Disparity"], data["PowerRank"])[0, 1]
print("Correlation:", round(correlation, 2))

# ==============================
# Create Figure with 2 Columns
# ==============================
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])

ax = fig.add_subplot(gs[0, 0])
legend_ax = fig.add_subplot(gs[0, 1])

# ==============================
# Scatter Plot With Red Circles and White Numbers
# ==============================
for i, row in data.iterrows():
    ax.scatter(
        row["Disparity"],
        row["PowerRank"],
        s=500,               # marker size
        facecolors='red',    # fill color
        edgecolors='black',  # border
        linewidth=1.5
    )
    # Place number inside circle = power ranking
    ax.text(
        row["Disparity"],
        row["PowerRank"],
        str(int(row["PowerRank"])),
        ha='center',
        va='center',
        fontsize=10,
        weight='bold',
        color='white'
    )

# ==============================
# Dynamic Line of Best Fit
# ==============================
m, b = np.polyfit(data["Disparity"], data["PowerRank"], 1)
x_vals = np.linspace(data["Disparity"].min(), data["Disparity"].max(), 100)
y_vals = m * x_vals + b
ax.plot(x_vals, y_vals, color='black', linewidth=2, label="Regression Line")

# Reference line at disparity = 1
ax.axvline(x=1, linestyle="--", color='gray', linewidth=1)

# Correlation display on plot
ax.text(0.05, 0.05, f"Correlation: {correlation:.2f}", transform=ax.transAxes,
        fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

# Formatting
ax.set_xlabel("Offensive Line Disparity (OL1 / OL2)")
ax.set_ylabel("Power Ranking (1 = Strongest)")
ax.set_title("Offensive Line Disparity vs Power Ranking")
ax.set_ylim(len(data)+0.5, 0.5)
ax.set_yticks(range(1, len(data)+1))

# ==============================
# Side Table: Number → Team
# ==============================
legend_ax.axis("off")
for i, row in data.iterrows():
    legend_ax.text(0, len(data) - i, f"{int(row['PowerRank'])}: {row['Team']}", va='center', fontsize=10)

legend_ax.set_ylim(0, len(data)+1)
legend_ax.set_xlim(0, 1)

plt.tight_layout()
plt.show()