
import matplotlib.pyplot as plt
import pandas as pd

# Data for comparison
data = {
    "Feature": ["Search Space", "Hardware", "Routing", "Performance", "Cost"],
    "Traditional RAG": ["Flat (693K chunks)", "High VRAM / Server", "LLM-based (Slow)", "\"Lost in Middle\"", "Cloud Fees"],
    "Microsoft GraphRAG": ["Recursive Summaries", "A100 / H100 GPU", "Global/Local Search", "Memory Intensive", "Enterprise Model"],
    "WiredBrain (Ours)": ["Hierarchical (99% Reduction)", "GTX 1650 (Laptop)", "3-Stage Neural (<50ms)", "Latency Optimized", "$0 (100% Local)"]
}

df = pd.DataFrame(data)

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

# Design colors
colors = [["#f2f2f2"] * 4, ["#ffffff"] * 4, ["#ffffff"] * 4, ["#ffffff"] * 4, ["#e6f7ff"] * 4]

# Create table
table = ax.table(cellText=df.values, 
                 colLabels=df.columns, 
                 cellLoc='center', 
                 loc='center',
                 colColours=["#333333"] * 4)

# Styling
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.5)

# Bold titles and colors
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
    if col == 3: # WiredBrain column
        cell.set_facecolor('#e6f7ff') # Soft light blue highlight
        if row == 0:
            cell.set_text_props(weight='bold', color='black') # Header text black
        else:
            cell.set_text_props(weight='bold')

plt.title("WiredBrain: Market Advantage Comparison", fontsize=16, pad=20, weight='bold')
plt.savefig('/home/user/Desktop/WiredBrain/WiredBrain-RAG/docs/images/market_comparison_clean.png', bbox_inches='tight', dpi=300)
print("Table image generated successfully.")
