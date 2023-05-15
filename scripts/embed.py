import asyncio
import pickle
from typing import List

import numpy as np
import openai
import pandas as pd
import torch.nn.functional as F
import tqdm
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

import src


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0
    )
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def embed_e5(sentences: List[str], model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    input_texts = ["passage: " + s for s in sentences]

    # Tokenize the input texts
    batch_dict = tokenizer(
        input_texts,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    model = model.to("cuda")
    outputs = model(**batch_dict.to("cuda"))
    embeddings = average_pool(
        outputs.last_hidden_state, batch_dict["attention_mask"]
    )

    # (Optionally) normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().detach().numpy().tolist()


async def dispatch_openai_requests(
    messages_list: List[str],
    model: str,
) -> List[str]:
    async_responses = []
    for message in messages_list:
        result = openai.Embedding.acreate(
            input=message,
            model=model,
        )
        embedding = result
        async_responses.append(embedding)
    return await asyncio.gather(*async_responses)


def get_ada_embeddings(sentences: List[str]):
    results = asyncio.run(
        dispatch_openai_requests(sentences, "text-embedding-ada-002")
    )
    return [x["data"][0]["embedding"] for x in results]


def embed(sentences, model_name: str):
    if model_name.startswith("hkunlp"):
        model = INSTRUCTOR(model_name)
        instructions = "Represent the science paragraph:"
        pair_list = [[instructions, s] for s in sentences]
        embeddings = model.encode(pair_list, show_progress_bar=True)
    elif model_name.startswith("intfloat"):
        embeddings = embed_e5(sentences, model_name)
    elif model_name.startswith("sentence-transformers"):
        model = SentenceTransformer(model_name)
        embeddings = model.encode(sentences, show_progress_bar=True)
    elif model_name == "text-embedding-ada-002":
        embeddings = get_ada_embeddings(sentences)
    else:
        raise NotImplementedError(
            "This model is not available yet: {model_name}."
        )

    return embeddings


def main():
    # Embed all the datasets.
    # Find out the names of the custom reranking matrices.
    datasets = ["sfn_2015_subsample.csv"]
    embeddings = src.embeddings.get_all_embeddings(
        "data/processed/hyperparameter_search.parquet"
    )
    for dataset in datasets:
        df = pd.read_csv("data/interim/" + dataset)
        for embedding in embeddings:
            refinement = ""
            if isinstance(embedding, list):
                embedding, refinement = embedding
                with open(
                    f"data/processed/{refinement}_matrix.pkl", "rb"
                ) as f:
                    matrix = pickle.load(f)
                print(matrix.shape)

            print(f"Embedding via {embedding}")
            strings_to_embed = (df.title + "\n" + df.abstract).values.tolist()
            if "intfloat" in embedding:
                chunk_size = 25
            else:
                chunk_size = 1000
            all_embeddings = []
            for i in tqdm.tqdm(range(0, len(strings_to_embed), chunk_size)):
                embeddings = embed(
                    strings_to_embed[i : i + chunk_size], embedding
                )
                all_embeddings.append(embeddings)
            embeddings = np.concatenate(all_embeddings, axis=0)
            if refinement:
                embeddings = embeddings @ matrix
            assert embeddings.shape[0] == len(strings_to_embed)
            with open(
                f"data/interim/{dataset.split('.')[0]}_{embedding.replace('/', '-')}{refinement}.pkl",
                "wb",
            ) as f:
                pickle.dump(embeddings, f)


if __name__ == "__main__":
    main()
