import requests
import tqdm
import json
import pandas as pd

url = "https://modeldb.science/metadata-predictor"

# Function to fetch a list of identifiers from a given API endpoint
def get_identifiers(api_endpoint):
    response = requests.get(api_endpoint)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error fetching identifiers from {api_endpoint}")

# Fetch lists of identifiers
celltypes_ids = get_identifiers("https://modeldb.science/api/v1/celltypes")
regions_ids = get_identifiers("https://modeldb.science/api/v1/regions")
currents_ids = get_identifiers("https://modeldb.science/api/v1/currents")
transmitters_ids = get_identifiers("https://modeldb.science/api/v1/transmitters")
concepts_ids = get_identifiers("https://modeldb.science/api/v1/modelconcepts")

# Convert identifiers to strings
celltypes_ids = [str(id) for id in celltypes_ids]
regions_ids = [str(id) for id in regions_ids]
currents_ids = [str(id) for id in currents_ids]
transmitters_ids = [str(id) for id in transmitters_ids]
concepts_ids = [str(id) for id in concepts_ids]




def get_info(abstract):
    data = {"abstract": abstract}

    # Categorize result_pairs into dictionaries
    categories = {
        "celltypes": [],
        "regions": [],
        "currents": [],
        "transmitters": [],
        "concepts": [],
    }

    response = requests.post(url, data=data)

    if response.status_code == 200:
        result_pairs = response.json()
        
        for obj_id, name in result_pairs:
            if str(obj_id) in celltypes_ids:
                categories["celltypes"].append(name)
            elif str(obj_id) in regions_ids:
                categories["regions"].append(name)
            elif str(obj_id) in currents_ids:
                categories["currents"].append(name)
            elif str(obj_id) in transmitters_ids:
                categories["transmitters"].append(name)
            elif str(obj_id) in concepts_ids:
                categories["concepts"].append(name)
    else:
        print(f"Error: {response.status_code} - {response.text}")



    # Create the final dictionary
    final_dict = {key: "; ".join(value) for key, value in categories.items()}

    return final_dict



with open("abstracts.json") as f:
    abstracts = json.load(f)

all_data = []
for pmid, pmid_data in tqdm.tqdm(abstracts.items()):
    abstract = []
    for item in pmid_data["AbstractText"]:
        abstract.append(item["text"])
    abstract = " ".join(abstract)
    data = get_info(abstract)
    data["pmid"] = int(pmid)
    all_data.append(data)

data = pd.DataFrame(all_data)
data.to_excel("rule_metadata.xlsx")