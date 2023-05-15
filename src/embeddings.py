import pandas as pd


def get_all_embeddings(hp_search=None):
    embeddings = [
        "text-embedding-ada-002",
        "hkunlp/instructor-large",
        "hkunlp/instructor-xl",
        "intfloat/e5-large",
        "intfloat/e5-base",
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/allenai-specter",
        "sentence-transformers/gtr-t5-large",
        "sentence-transformers/all-mpnet-base-v2",
    ]
    if hp_search is not None:
        df_hp = pd.read_parquet(hp_search)
        df_hp.run_id.unique().values.tolist()
        embeddings += [
            ["sentence-transformers/all-mpnet-base-v2", x]
            for x in df_hp.run_id.unique().values.tolist()
        ]

    return embeddings
