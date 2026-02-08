
import matplotlib.pyplot as plt
import pandas as pd

# Updated Data for comparison with Microsoft Solving points
data = {
    "Technical Metric": [
        "Search Space", 
        "Hardware Required", 
        "MS Constraint (Lost-in-Middle)", 
        "Reasoning Path", 
        "Deployment Cost"
    ],
    "Traditional RAG": [
        "Flat (693K chunks)", 
        "High VRAM / Server", 
        "Vulnerable (Accuracy Drop)", 
        "Black-Box (Stochastic)", 
        "Cloud API Fees"
    ],
    "Microsoft GraphRAG": [
        "Recursive Summaries", 
        "Enterprise (A100/H100)", 
        "Summarization Overload", 
        "LLM-Intensive", 
        "Premium Licensing"
    ],
    "WiredBrain (Ours)": [
        "Hierarchical (99% Reduction)", 
        "Consumer (GTX 1650)", 
        "SOLVED: Address Routing", 
        "Transparent (X/Y/Z Streams)", 
        "$0 (100% Local)"
    ]
}

df = pd.DataFrame(data)

# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 7))
ax.axis('off')

# Design colors
header_color = "#1a1a1a"
wiredbrain_color = "#e6f7ff"
font_family = 'sans-serif'

# Create table
table = ax.table(cellText=df.values, 
                 colLabels=df.columns, 
                 cellLoc='left', 
                 loc='center',
                 colColours=[header_color] * len(df.columns))

# Styling
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.1, 2.8)

# Stylistic refinements
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#d9d9d9')
    cell.set_linewidth(0.5)
    
    # Header Styling
    if row == 0:
        cell.set_text_props(weight='bold', color='white', family=font_family)
        cell.set_facecolor(header_color)
        if col == 3: # WiredBrain column header
            cell.set_facecolor(wiredbrain_color)
            cell.set_text_props(weight='bold', color='black')
    
    # WiredBrain Column Highlight
    if col == 3 and row > 0:
        cell.set_facecolor(wiredbrain_color)
        cell.set_text_props(weight='bold', color='#004085') # Deep blue for emphasis
        
    # Microsoft Solve Highlight
    if row == 3 and col == 3:
        cell.set_text_props(color='#d32f2f') # Red highlight for "SOLVED"

plt.title("WiredBrain: Dominating the Consumer RAG Market", fontsize=18, pad=30, weight='bold', family=font_family)
plt.savefig('/home/user/Desktop/WiredBrain/WiredBrain-RAG/docs/images/market_comparison_clean.png', bbox_inches='tight', dpi=300)
print("Professional Table image with Microsoft Solve points generated successfully.")
