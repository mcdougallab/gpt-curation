import json
import tqdm
import numpy as np
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from statistics import mean
from scipy.spatial import distance
from matplotlib import pyplot as plt


#getting data
with open('embeddings_model.json') as json_file:
    embeddings_model = json.load(json_file)
with open('embeddings_no_model(2).json') as json_file:
    embeddings_no_model = json.load(json_file)


#get cosine similarity score between two embeddings
def get_cosine_sim(a,b):
  return distance.cosine(a,b)


# get k nearest neighbors and longest distance among NNs
def get_knns(k):
  print("Training -------------------")
  #train NN models, both positive and negative
  NN_model = NearestNeighbors(n_neighbors = k+1, metric = 'cosine', n_jobs = -1)
  NN_model.fit(embeddings_model)

  #get information of k NNs of models
  print("Retrieving Positive -------------------")
  NN_info_pos = {}
  max_distances_pos = []

  for i in tqdm.tqdm(range(len(embeddings_model))):
      dist, inds = NN_model.kneighbors([embeddings_model[i]], return_distance=True)
      avg = mean(dist[0][1:])
      max_d = max(dist[0][1:])
      results = {"avg_dist": avg, "max_dist": max_d, "distances": dist[0][1:].tolist(), "indexes": inds[0][1:].tolist()}

      NN_info_pos[str(i)] = results
      max_distances_pos.append(max_d)


  #get information of k NNs of non models
  print("Retrieving Negative -------------------")
  NN_info_neg = {}
  max_distances_neg = []
  for i in tqdm.tqdm(range(len(embeddings_no_model))):
    dist, inds = NN_model.kneighbors([embeddings_no_model[i]], return_distance=True)
    avg = mean(dist[0][1:])
    max_d = max(dist[0][1:])
    results = {"avg_dist": avg, "max_dist": max_d, "distances": dist[0][1:].tolist(), "indexes": inds[0][1:].tolist()}

    NN_info_neg[str(i)] = results
    max_distances_neg.append(max_d)

  return NN_info_pos, max_distances_pos, NN_info_neg, max_distances_neg



#distances to test
distances = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30]

def get_fractions(NN_info_pos, max_distances_pos, NN_info_neg, max_distances_neg):
   
  #calculate A
  A_info = []
  for d in distances:
    A_counter = 0
    for i in range(len(max_distances_pos)):
        if max_distances_pos[i] < d:
          A_counter += 1
    A_info.append(A_counter/1564)
  
  #calculate B
  B_info = []
  for d in distances:
    B_counter = 0
    for i in range(len(max_distances_neg)):
        if max_distances_neg[i] < d:
          B_counter += 1
    B_info.append(B_counter/1700)
  
  return A_info, B_info
      


#plotting
fig, axs = plt.subplots(3, figsize=(6,8))
axs[0].set_title("Fraction of MODELS that have their kth NN within distance d", fontsize = 16)
axs[1].set_title("Fraction of NON MODELS that have their kth NN within distance d", fontsize = 16)
axs[2].set_title("Difference between (A) and (B)", fontsize = 16)

plt.subplots_adjust(hspace=0.8, right=0.75)

#for different values of k
for k in [5, 10, 20, 50, 100, 500, 1000, 1564]:
  print("For k = " + str(k) + " -------------------")
  NN_info_pos, max_distances_pos, NN_info_neg, max_distances_neg = get_knns(k)
  A_info, B_info = get_fractions(NN_info_pos, max_distances_pos, NN_info_neg, max_distances_neg)
  axs[0].plot(distances, A_info, label = "K = " + str(k))
  axs[1].plot(distances, B_info, label = "K = " + str(k)) 
  axs[2].plot(distances, list(np.array(A_info) - np.array(B_info)), label = "K = " + str(k)) 
  
axs[1].legend(fontsize = 16, loc='upper left', bbox_to_anchor=(1, 1))


#edit axis values size
for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=16)
plt.show()
