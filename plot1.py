import matplotlib.pyplot as plt
import numpy as np
metrics = ['Precision', 'Coverage', 'Accuracy', 'Absolute true', '1 - Absolute false']
models_data = {
    'PrMFTP': [0.699, 0.669, 0.651, 0.593, 1 - 0.031],
    'ETFC': [0.724, 0.717, 0.684, 0.617, 1 - 0.036],
    'CELA-MFP': [0.739, 0.754, 0.700, 0.611, 1 - 0.032],
    'iMFP-LG': [0.730, 0.730, 0.689, 0.616, 1 - 0.032],
    'PSCFA': [0.744, 0.717, 0.696, 0.637, 1 - 0.034],
    'MCMFPP': [0.770, 0.741, 0.721, 0.660, 1 - 0.031]
}

colors = ['#ff9c4d', '#999999', '#a6cee3', '#b08fc7', '#ffd966', '#1f77b4']
bar_width = 0.13
index = np.arange(len(metrics))
plt.figure(figsize=(12, 6))
for i, (model, values) in enumerate(models_data.items()):
    r = index + bar_width * i
    bars = plt.bar(r, values, width=bar_width, label=model, color=colors[i])
    for j, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f'{values[j]:.3f}', ha='center', va='bottom', rotation=90)
plt.xticks(index + bar_width * (len(models_data) - 1) / 2, metrics)
plt.ylabel('Performance Value')
plt.ylim(0.5, 1.05)
plt.legend(loc='upper left')
plt.savefig('Figures/model_evaluation.jpg', dpi=500)
plt.show()