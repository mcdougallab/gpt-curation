import sqlite3
import pandas as pd
import json
import openai
import time
import html

openai.api_key = "--REDACTED-PUT-YOUR-API-KEY-HERE--"
MY_DB = "paper_data_cot.db"


OUT_LOG = "query_openai.log"


#with open(OUT_LOG, "w") as f:
#    ...


def make_query(title: str, abstract: str)->str:
    return f"""You are an expert in computational neursocience, reviewing papers for possible inclusion in a repository of computational neuroscience. This database includes papers that use computational models written in any programming language for any tool, but they all must have a mechanistic component for getting insight into the function individual neurons, networks of neurons, or of the nervous system in health or disease.
    
    Suppose that a paper has title and abstract as indicated below. Perform the following steps, numbering each answer as follows.
    
    1. Identify evidence that this paper addresses a problem related to neuroscience or neurology.

    2. Identify specific evidence in the abstract that directly suggests the paper uses computational neuroscience approaches (e.g. simulation with a mechanistic model). Do not speculate beyond the methods explicitly mentioned in the abstract.

    3. Identify evidence that this paper uses machine learning or any other computational methods.

    4. Provide a one-word final assessment of either "yes", "no", or "unsure" (include the quotes but provide no other output) as follows: Respond with "yes" if the paper likely uses computational neuroscience approaches (e.g. simulation with a mechanistic model), and "no" otherwise. Respond "unsure" if it is unclear if the paper uses a computational model or not. In particular, respond "yes" for a paper that uses both computational neuroscience and other approaches. Respond "no" for a paper that uses machine learning to make predictions about the nervous system but does not include a mechanistic model. Respond "no" for purely experimental papers. Provide no other output.
    
    Title: "{title}"
    
    Abstract: "{abstract}" """


total_tokens = 0


def query_openai(pmid: int, title: str, abstract: str, distance: float):
    global total_tokens
    query = make_query(html.unescape(title), html.unescape(abstract))
    data = {
        "pmid": [pmid],
        "distance": [distance],
        "title": [title],
    }
    for attempt in [1, 2, 3]:
        column = f"cot_relevant35_attempt_{attempt}"
        sleep_delay = 1
        while True:
            time.sleep(sleep_delay)
            sleep_delay *= 1.5
            if sleep_delay > 120:
                sleep_delay = 100
            try:    
                completions = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": query}],
                    max_tokens=2_000,
                )
            except openai.error.ServiceUnavailableError:
                print("failed... retrying in", sleep_delay)
                continue
            except openai.error.RateLimitError:
                sleep_delay *= 2
                print("failed... RateLimitError... retrying in", sleep_delay)
                continue
            except openai.error.APIError:
                sleep_delay = 100
                print("failed.... APIError... retrying in", sleep_delay)
                continue
            except openai.error.Timeout:
                sleep_delay = 200
                print("failed.... TimeOut... retrying in", sleep_delay)
                continue
            except:
                sleep_delay = 1_000
                print("unknown failure... trying again in 1_000 sec")
                continue
            break
        response = completions.choices[0].message.content
        data[column] = [response]
        with open(OUT_LOG, "a") as f:
            f.write(json.dumps({"pmid": pmid, "response": response}) + "\n")
        total_tokens += completions["usage"]["total_tokens"]
        print("tokens", total_tokens)
    with sqlite3.connect(MY_DB) as conn:
        pd.DataFrame(data).to_sql("relevant", conn, if_exists="append", index=False)


all_data = pd.read_csv("../20230630/results/test_k5.csv").sort_values("distance k=5")

try:
    with sqlite3.connect(MY_DB) as conn:
        known_pmids = set(pd.read_sql("SELECT pmid FROM relevant", conn).pmid.to_list())
except pd.io.sql.DatabaseError:
    known_pmids = set()

for i, (_, data) in enumerate(all_data.iterrows()):
    pmid = data["pmid"]
    if pmid not in known_pmids:
        metadata = eval(data["metadata"])[1]
        distance = data["distance k=5"]
        title = metadata["ArticleTitle"]
        abstract = metadata["AbstractText"]
        print(pmid)
        query_openai(pmid, title, abstract, distance)
        if i > 100:
            break

