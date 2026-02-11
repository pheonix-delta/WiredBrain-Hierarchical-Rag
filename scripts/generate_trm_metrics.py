
import matplotlib.pyplot as plt
import numpy as np

# Data for TRM Efficiency
iterations = np.array([1, 2, 3, 4, 5])
confidence = np.array([0.45, 0.68, 0.82, 0.94, 0.98])  # Confidence growth
hallucination_rate = np.array([0.22, 0.12, 0.05, 0.02, 0.01]) # Internal detection drop

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Reasoning Iterations (TRM Steps)')
ax1.set_ylabel('Confidence Score (Gaussian Check)', color=color)
ax1.plot(iterations, confidence, color=color, marker='o', linewidth=3, label='Confidence')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('Hallucination Rate (Simulated Detection %)', color=color)
ax2.plot(iterations, hallucination_rate, color=color, marker='x', linestyle='--', linewidth=3, label='Hallucination Rate')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  
plt.title('Transparent Reasoning Module (TRM) Performance Metrics', fontsize=14, weight='bold', pad=20)
plt.savefig('/home/user/Desktop/WiredBrain/WiredBrain-RAG/docs/images/fig9_trm_metrics.png', dpi=300, bbox_inches='tight')
print("TRM Metrics chart generated successfully: fig9_trm_metrics.png")
