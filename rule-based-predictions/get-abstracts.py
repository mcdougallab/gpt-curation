import time
import tqdm
import json
import requests
import xml.etree.ElementTree as ET

pmids = [35500112, 34894291, 35552409, 35245281, 35569510, 35031915, 35613586, 35916367, 35180461, 35140075, 36344267, 35998146, 35750111, 35995031, 36034337, 34398769, 36577383, 36067727, 35165198, 35441302, 35041661, 35971033, 35320026, 35181595, 35079038, 35413446, 36058524, 36228575, 36323513, 36396092, 36099807, 35364840, 35526721, 35136991, 36087422, 36302178, 36456345, 35124593, 35417922, 35170022, 34978378, 36008421, 34503414, 35364826, 35970021, 36329492, 34911427, 35149118, 35538324, 35104646, 36310030, 36327232, 36086302, 36526822, 35771657, 36395336, 36056083, 36018837, 35346837, 34431994, 36516780, 35306174, 35728405, 35879779, 36410634, 36362443, 36086062, 36197012, 35257321, 36327349, 35680927, 35835362, 35272023, 35786442, 35568311, 35838898, 34686937, 34665396, 35850254, 35037686, 35913987, 34713909, 34837245, 36412516, 35325849, 35921960, 35034741, 34942160, 35141747, 35058372, 34806782, 35502706, 35772526, 34880404, 35976323, 36374001, 35793632, 36536272, 35523580, 35417276, 35820645, 36562479, 35580809, 35840120, 35569783, 36229510, 35508164, 35690132, 35104819, 35702056, 36538267, 35490971, 35637276, 35883609, 36322525]


def lookup_pmids(pmids, delay_interval=1):
    time.sleep(delay_interval)
    return ET.fromstring(
        requests.post(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            data={
                "db": "pubmed",
                "retmode": "xml",
                "id": ",".join(f"{pmid}" for pmid in pmids),
            },
        ).text
    )


def parse_paper(paper):
    abstract_block = paper.find(".//Abstract")
    try:
        pmid = int(paper.find(".//PMID").text)
    except AttributeError:
        raise Exception("Bad paper? " + ET.tostring(paper, method="text").decode())
    title = paper.find(".//ArticleTitle")
    if title is None:
        title = paper.find(".//BookTitle")

    if abstract_block is not None:
        abstract = [
            {
                "section": item.get("Label"),
                "text": ET.tostring(item, method="text").decode(),
            }
            for item in abstract_block
            if item.tag == "AbstractText"
        ]
    else:
        abstract = ""
    assert title is not None
    title = ET.tostring(title, method="text").decode()

    return pmid, {"AbstractText": abstract, "ArticleTitle": title}


def process_pmids(pmids, delay_interval=1):
    results = {}
    papers = lookup_pmids(pmids, delay_interval=delay_interval)
    for paper in tqdm.tqdm(papers):
        pmid, parsed_paper = parse_paper(paper)
        results[pmid] = parsed_paper
    return results



data = process_pmids(pmids)

with open("abstracts.json", "w") as f:
    json.dump(data, f, indent=4)