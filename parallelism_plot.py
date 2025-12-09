import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


data = {
    "HGP": [
        {"name": "[[225,9,6]]", "serial": 1512, "parallel": 8},
        {"name": "[[375,15,6]]", "serial": 2520, "parallel": 10},
        {"name": "[[625,25,8]]", "serial": 4200, "parallel": 12}
    ],
    "BB": [
        {"name": "[[72,12,6]]", "serial": 432, "parallel": 12},
        {"name": "[[90,8,10]]", "serial": 540, "parallel": 12},
        {"name": "[[144,12,12]]", "serial": 864, "parallel": 12}
    ],
}


labels = []
speedups = []
colors = []
category_boundaries = []
current_index = 0

for category, codes in data.items():
    for code in codes:
        labels.append(code['name'])
        speedups.append(code['serial'] / code['parallel'])
        colors.append('#32CD32' if category == "HGP" else '#40E0D0') 
    current_index += len(codes)
    category_boundaries.append(current_index)


fig, ax = plt.subplots(figsize=(12, 6))
bar_positions = np.arange(len(speedups))
bars = ax.bar(bar_positions, speedups, color=colors)

css_start = category_boundaries[0]
css_end = category_boundaries[1]
ax.axvspan(css_start - 0.5, css_end - 0.5, color='#e0e0e0', alpha=0.4, hatch='//', edgecolor='gray', linewidth=0)


ax.set_xticks(bar_positions)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel('Speedup Over Full Serialization')
ax.set_yscale("log")
ax.set_yticks([1, 10, 100, 1000], [1, 10, 100, 1000])


category_names = list(data.keys())
start = 0
for i, end in enumerate(category_boundaries):
    midpoint = (start + end - 1) / 2
    ax.text(midpoint, max(speedups) * 0.93, category_names[i], ha='center', fontsize=10.5, weight='bold')
    start = end

plt.tight_layout()
plt.gcf().set_size_inches(5.0, 2.5)
plt.savefig("ParallelismMotivation.png", dpi=300, bbox_inches="tight")
plt.show()
