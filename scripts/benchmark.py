import pickle

import numpy as np
import pandas as pd
import tqdm
from sentence_transformers import util

import src


def calculate_overlap(t1, t2):
    parts1 = t1.split(".")[:3]
    parts2 = t2.split(".")[:3]
    overlap = 0
    for p0, p1 in zip(parts1, parts2):
        if p0 == p1:
            overlap += 1
        else:
            break

    return overlap


def get_top(df, embeddings):
    tops = []

    for search_id in range(df.shape[0]):
        doc = df.iloc[search_id]
        search_hits = util.semantic_search(
            embeddings[search_id], embeddings, top_k=11
        )[0][1:]
        # Take the top 5 hits

        # Calculate tag overlap
        retrieval = {
            "probe_id": doc.id,
            "probe_title": doc.title,
            "probe_abstract": doc.abstract,
        }

        # Use topic overlap
        topic = doc.topic
        overlap = 0
        for i, t in enumerate(search_hits):
            doc2 = df.iloc[t["corpus_id"]]
            topic2 = doc2["topic"]
            overlap += calculate_overlap(topic, topic2)
            retrieval[f"id_{i}"] = doc2.id
            retrieval[f"title_{i}"] = doc2.title
            retrieval[f"abstract_{i}"] = doc2.abstract

            if i in (0, 4, 9):
                retrieval[f"score_{i+1}"] = overlap / (i + 1)

        tops.append(retrieval)

    return tops


def main():
    # Embed all the datasets.
    datasets = ["sfn_2015_subsample.csv"]
    embeddings = src.embeddings.get_all_embeddings(
        "data/processed/hyperparameter_search.parquet"
    )
    dfs = []
    for dataset in datasets:
        df = pd.read_csv("data/interim/" + dataset)
        for embedding in tqdm.tqdm(embeddings):
            refinement = ""
            if isinstance(embedding, list):
                embedding, refinement = embedding

            fname = f"data/interim/{dataset.split('.')[0]}_{embedding.replace('/', '-')}{refinement}.pkl"
            with open(fname, "rb") as f:
                embeddings = pickle.load(f)
            embeddings = np.array(embeddings)

            scores = get_top(df, embeddings)
            df_scores = pd.DataFrame(scores)
            df_scores["dataset"] = dataset
            df_scores["embedding"] = embedding
            df_scores["refinement"] = refinement
            dfs.append(df_scores)

    df = pd.concat(dfs, axis=0)
    df.to_csv("data/processed/scores.csv", index=False)


if __name__ == "__main__":
    main()
