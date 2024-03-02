from sklearn.metrics import cohen_kappa_score
import pandas

#k=5 scores_matrix

k5 = pandas.read_csv("./k5_mergedupdated.csv")

k5['Kaylin'] = k5['Kaylin'].map({'T':int(1), 'F':int(0)})

k5['Amy'] = k5['Amy'].map({'T':int(1), 'F':int(0)})

k5 = k5.iloc[:100,:]

amy = k5['Amy'].tolist()

kay = k5['Kaylin'].astype(int).tolist()

k5['gpt3.5_relevant'] = k5['gpt3.5_relevant'].str.lower()

gpt35 = k5['gpt3.5_relevant'].map({'yes':int(1), 'no':int(0)}).tolist()

k5['gpt4_relevant'] = k5['gpt4_relevant'].str.lower()

gpt4 = k5['gpt4_relevant'].map({'yes':int(1), 'no':int(0)}).tolist()

all_lists = [amy, kay, gpt35, gpt4]

scores_matrix = [[0]*len(all_lists) for _ in range(len(all_lists))]

for i in range(4):
    for j in range(4):
        score = cohen_kappa_score(all_lists[i], all_lists[j])
        scores_matrix[i][j] = score

k5matrix = scores_matrix

#k=20 scores_matrix

k20 = pandas.read_csv("./k20_mergedupdated.csv")

k20 = k20.iloc[:100,:]

k20['gpt3.5_relevant'] = k20['gpt3.5_relevant'].str.lower()

gpt35 = k20['gpt3.5_relevant'].map({'yes':int(1), 'no':int(0)}).tolist()

k20['gpt4_relevant'] = k20['gpt4_relevant'].str.lower()

gpt4 = k20['gpt4_relevant'].map({'yes':int(1), 'no':int(0)}).tolist()

k20['Kaylin'] = k20['Kaylin'].str.lower()

kaylin = k20['Kaylin'].map({'yes':int(1), 'no':int(0)}).tolist()

k20['Amy'] = k20['Amy'].str.lower()

amy = k20['Amy'].map({'yes':int(1), 'no':int(0)}).tolist()

all_lists = [amy, kaylin, gpt35, gpt4]

k20scores_matrix = [[0]*len(all_lists) for _ in range(len(all_lists))]

for i in range(4):
    for j in range(4):
        score = cohen_kappa_score(all_lists[i], all_lists[j])
        k20scores_matrix[i][j] = score

#k=50 scores_matrix

k50 = pandas.read_csv("./k50_mergedupdated.csv")

k50 = k50.iloc[:100,:]

amy = k50['''Amy's'''].tolist()
kaylin = k50['Kaylin'].tolist()
gpt4_relevant = k50['gpt4_relevant'].tolist()
gpt35_relevant = k50['gpt3.5_relevant'].tolist()

all_lists = [amy, kaylin, gpt35_relevant, gpt4_relevant]

k50scores_matrix = [[0]*len(all_lists) for _ in range(len(all_lists))]

for i in range(4):
    for j in range(4):
        score = cohen_kappa_score(all_lists[i], all_lists[j])
        k50scores_matrix[i][j] = score

import matplotlib.pyplot as plt
import seaborn as sns


fig, axes = plt.subplots(1, 3, figsize=(11.11, 4.34), sharey=True)

sns.heatmap(k5matrix, annot=True, fmt=".3f", ax=axes[0], cbar=False, cmap='viridis')
sns.heatmap(k20scores_matrix, annot=True, fmt=".3f", ax=axes[1], cbar=False, cmap='viridis')
heatmap3 = sns.heatmap(k50scores_matrix, annot=True, fmt=".3f", ax=axes[2], cbar=True, cbar_kws={'label': 'Agreement Scores'}, cmap='viridis')
fig.suptitle("Cohen Kappa's Agreement", fontsize=20)
cbar = heatmap3.collections[0].colorbar
cbar.ax.tick_params(labelsize=10)
cbar.set_label('Agreement Scores', size=14) 

axes[0].set_title('k = 5', fontsize=14)
axes[1].set_title('k = 20', fontsize=14)
axes[2].set_title('k = 50', fontsize=14)

axes[0].set_xlabel('Annotator', fontsize=11)
axes[1].set_xlabel('Annotator', fontsize=11)
axes[2].set_xlabel('Annotator', fontsize=11)

axes[0].set_ylabel('Annotator', fontsize=11)

for ax in axes:
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_xticklabels(['S1', 'S2', 'gpt3.5', 'gpt4'], fontsize = 13)
    ax.set_yticklabels(['S1', 'S2', 'gpt3.5', 'gpt4'], fontsize = 13)

plt.tight_layout()

plt.show()
