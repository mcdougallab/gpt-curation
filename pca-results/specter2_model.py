import time
import tqdm
import json
import requests
import xml.etree.ElementTree as ET
from transformers import AutoTokenizer, AutoModel



#read in the ids of papers with models
with open('meta_data_model.json') as json_file:
    metadata = json.load(json_file)


# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('allenai/specter2')
model = AutoModel.from_pretrained('allenai/specter2')


#create function that creates embeddings given a dictionary of papers
def create_embeddings(papers):
  embeddings = {}
  for pmid, paper in tqdm.tqdm(papers.items()):
      data = [paper["ArticleTitle"] + tokenizer.sep_token + paper["AbstractText"]]
      inputs = tokenizer(
          data, padding=True, truncation=True, return_tensors="pt", max_length=512
      )
      result = model(**inputs)
      # take the first token in the batch as the embedding
      embeddings[pmid] = result.last_hidden_state[:, 0, :].detach().numpy()[0]

  # turn our dictionary into a list
  embeddings = [embeddings[pmid] for pmid in papers.keys()]
  return embeddings



#create embeddings
embeddings = create_embeddings(metadata)

#convert ndarray to list
embeddings_temp = []
for embedding in embeddings:
  embeddings_temp.append(embedding.tolist())

#save embeddings as json files
with open("test_embeddings_model.json", "w") as outfile:
    json.dump(embeddings_temp, outfile)