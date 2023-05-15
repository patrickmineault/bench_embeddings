"""
Compile datasets necessary for re-ranking SFN abstract.

We need two datasets:

1. All the abstracts in the SFN abstract book embedded using a base embedding.
2. Positive and negative pairs for re-ranking.
"""
import pickle

import pandas as pd
from sentence_transformers import SentenceTransformer


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


def main():
    dataset = "sfn_2015.csv"
    embedding = "sentence-transformers/all-mpnet-base-v2"
    df = pd.read_csv(f"data/raw/{dataset}")
    sz0 = df.shape[0]
    df_leave_aside = pd.read_csv("data/interim/sfn_2015_subsample.csv")

    ids = df_leave_aside["id"].values.tolist()
    df = df[~df["id"].isin(ids)]
    df["text"] = df.title + "\n" + df.abstract
    df.to_parquet("data/interim/sfn_2015_train.parquet", index=False)

    # Embed the data.
    # Load the embedding model.
    # Now, let's embed the data.
    model = SentenceTransformer(embedding)
    embeddings = model.encode(df.text.values, show_progress_bar=True)
    # Save the embeddings.
    fname = f"data/interim/{dataset.split('.')[0]}_{embedding.replace('/', '-')}_all.pkl"
    with open(fname, "wb") as f:
        pickle.dump(embeddings, f)

    df["embedding"] = embeddings.tolist()

    assert df.shape[0] == sz0 - df_leave_aside.shape[0]

    # Now compile positive and negative pairs.
    # To start with, use a ratio of 1:1 between positive and negative pairs.
    # Mine hard negatives (correct up to the second leaf element).
    level_0 = df.topic.map(lambda x: x.split(".")[0])
    level_1 = df.topic.map(lambda x: ".".join(x.split(".")[:2]))
    level_2 = df.topic.map(lambda x: ".".join(x.split(".")[:3]))

    df["level_0"] = level_0
    df["level_1"] = level_1
    df["level_2"] = level_2

    pairs = []

    df_indexed = df.set_index("id")

    # Select a 10% sample of the data as belonging to the test set.
    test_ids = df.sample(frac=0.1).id.values.tolist()

    for i, row in df.iterrows():
        l2_pos = df[(df.level_2 == row.level_2) & (df.id > row.id)]
        l1_pos = df[
            (df.level_1 == row.level_1)
            & (df.level_2 != row.level_2)
            & (df.id > row.id)
        ]
        l0_pos = df[
            (df.level_0 == row.level_0)
            & (df.level_1 != row.level_1)
            & (df.level_2 != row.level_2)
            & (df.id > row.id)
        ]
        neg = df[
            (df.level_0 != row.level_0)
            & (df.level_1 != row.level_1)
            & (df.level_2 != row.level_2)
            & (df.id > row.id)
        ]

        if (
            len(l2_pos) < 3
            or len(l1_pos) < len(l2_pos)
            or len(l0_pos) < len(l2_pos)
            or len(neg) < len(l2_pos)
        ):
            continue

        # All the positive pairs are acceptable, but we subsample negatives
        for label, examples in [
            (1, l2_pos),
            (0.67, l1_pos.sample(n=len(l2_pos) // 3)),
            (0.33, l0_pos.sample(n=len(l2_pos) // 3)),
            (0, neg.sample(n=len(l2_pos) // 3)),
        ]:
            pairs += [
                {
                    "id_1": row.id,
                    "id_2": x,
                    "label": label,
                    "text_1": row.text,
                    "text_1_embedding": row.embedding,
                    "text_2": df_indexed.loc[x].text,
                    "text_2_embedding": df_indexed.loc[x].embedding,
                    "fold": "test"
                    if (x in test_ids or row.id in test_ids)
                    else "train",
                }
                for x in examples.id.values.tolist()
            ]

    # Save the pairs
    df_pairs = pd.DataFrame(pairs)
    df_pairs.to_parquet("data/interim/sfn_2015_pairs.parquet", index=False)


if __name__ == "__main__":
    main()
