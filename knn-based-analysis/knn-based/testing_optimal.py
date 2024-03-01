import json
import tqdm
import numpy as np
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from statistics import mean
from scipy.spatial import distance
from matplotlib import pyplot as plt
import pandas as pd



#optimal distances from k=5 and k=100
k_5 = [0.0, 0.0, 0.0, 0.01278772378516624, 0.16687979539641945, 0.4827877237851662, 0.7539386189258311, 0.8931969309462915, 0.936777493606138, 0.9081841432225064, 0.8145012787723784, 0.638772378516624, 0.4294629156010231, 0.2424040920716114, 0.12887468030690552, 0.047698209718670226, 0.016521739130434865, 0.008286445012787858, 0.0012276214833760735, 0.0012276214833760735, 0.0006393861892584063, 0.0006393861892584063, 0.0006393861892584063, 0.0006393861892584063, 0.0006393861892584063, 0.0006393861892584063, 0.0006393861892584063, 0.0006393861892584063, 0.0006393861892584063, 0.0006393861892584063]
k_100 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.004475703324808184, 0.09335038363171355, 0.35171355498721224, 0.6375191815856778, 0.8059335038363171, 0.9037851662404092, 0.9288746803069053, 0.8863938618925831, 0.7615601023017903, 0.5717135549872123, 0.36705882352941177, 0.20063938618925836, 0.08299232736572904, 0.034168797953964325, 0.01181585677749375, 0.0018158567774937406, 0.0018158567774937406, 0.0006393861892584063, 0.0006393861892584063, 0.0006393861892584063, 0.0006393861892584063, 0.0006393861892584063, 0.0006393861892584063, 0.0006393861892584063, 0.0006393861892584063]
distances = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30]

d_5 = distances[k_5.index(max(k_5))] #optimal distance 
d_100 = distances[k_100.index(max(k_100))] #optimal distance 0.12 k= 100


#getting embeddings of known models
with open('embeddings_model.json') as json_file:
    embeddings_model = json.load(json_file)
#getting test pmids
with open('test_ids.json') as json_file:
    test_ids = json.load(json_file)
#getting test metadata
with open('test_metadata.json') as json_file:
    test_metadata = json.load(json_file)

test_metadata = [str(item) for item in test_metadata.items()]

print("Retrieving embeddings -------------------")
#getting test embeddings
with open('test_embeddings1-specter2.json') as json_file:
    test_embeddings1 = json.load(json_file)
with open('test_embeddings2-specter2.json') as json_file:
    test_embeddings2 = json.load(json_file)
with open('test_embeddings3-specter2.json') as json_file:
    test_embeddings3 = json.load(json_file)

test_embeddings = test_embeddings1 + test_embeddings2 + test_embeddings3

# get k nearest neighbors and longest distance among NNs
def get_knns(k):
  print("Training -------------------")
  #train NN models, both positive and negative
  NN_model = NearestNeighbors(n_neighbors = k, metric = 'cosine', n_jobs = -1)
  NN_model.fit(embeddings_model)

  #get information of k NNs of models
  print("Retrieving Positive -------------------")
  NN_info_test = {}
  max_distances_test = []

  for i in tqdm.tqdm(range(len(test_embeddings))):
      dist, inds = NN_model.kneighbors([test_embeddings[i]], return_distance=True)
      avg = mean(dist[0][:])
      max_d = max(dist[0][:])
      results = {"avg_dist": avg, "max_dist": max_d, "distances": dist[0][:].tolist(), "indexes": inds[0][:].tolist()}

      NN_info_test[str(i)] = results
      max_distances_test.append(max_d)

  return NN_info_test, max_distances_test



# k
NN_info_test, max_distances_test = get_knns(20)

data = {"pmid": test_ids, "distance k=5": max_distances_test, "metadata": test_metadata}
df = pd.DataFrame(data)
df.to_csv("test_k20.csv")

