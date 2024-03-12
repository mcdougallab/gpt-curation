import matplotlib.pyplot as plt
import numpy as np

categories = ['C', 'B', 'I']
titles = ['Concepts', 'Regions', 'Ion Channels', 'Cell Types', 'Receptors', 'Transmitters']

results_gpt4 = [[732, 9, 12], [199, 3, 3], [39, 0, 6], [95, 6, 34], [27, 4, 3], [41, 2, 1]]
results_rule_based = [[386, 3, 14], [113, 0, 4], [27, 1, 3], [42, 1, 14], [19, 0, 1], [27, 0, 1]]

colors = ['Red', 'Blue', 'Green', 'Purple', 'Pink', 'Yellow']

def add_labels(ax, data, height_factor=1.02, fontsize=12):
    for i, val in enumerate(data):
        percentage = val / sum(data) * 100
        ax.text(i, val * height_factor, f'{percentage:.1f}%', ha='center', color='black', fontsize=fontsize)

fig, axs = plt.subplots(2, 6, figsize=(20, 13))

fig.suptitle('Metadata Prediction Model Comparison: GPT-4 vs Rule-Based', fontsize=24, y=1)

for index_row, (row, result, method) in enumerate(zip(axs, [results_gpt4, results_rule_based], ['GPT-4', 'Rule-Based'])):
    for index_ax, (ax, counts, color, title) in enumerate(zip(row, result, colors, titles)):
        ax.bar(categories, counts, color=color)
        add_labels(ax, counts, fontsize=14)
        ax.set_title(title, fontsize=17)
        ax.set_ylim(0, max(counts) * 1.2)
        ax.tick_params(axis='x', labelsize=14)
        ax.set_ylabel('Number of Tags', fontsize=17)
        ax.legend([f'Total: {sum(counts)}'], loc='upper right', fontsize=13)
        if index_ax == 0:
            ax.annotate(method, xy=(-0.4, 0), xytext=(-0.3, 0.5),
                        textcoords='axes fraction', ha='right', va='center', 
                        fontsize=21, rotation='vertical')

plt.savefig('./Metadata Prediction Comparison')
plt.show()
