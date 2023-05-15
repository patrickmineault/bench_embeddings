# Benchmark NLP embeddings.

Benchmark NLP embeddings for science retrieval. Uses the SFN 2015 dataset from [Science Concierge: A Fast Content-Based Recommendation System for Scientific Publications](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0158423). Tries many off-the-shelf embeddings and also trains a linear projection on top of the embeddings. The training script for training the linear projection is adapted from [this sample script from OpenAI](https://github.com/openai/openai-cookbook/blob/main/examples/Customizing_embeddings.ipynb).

The TL;DR is that:

1. The off-the-shelf sentence transformer from sbert performs *better* than OpenAI's embeddings. This is perhaps surprising comparing to the [results of MTEB](https://huggingface.co/spaces/mteb/leaderboard), but it seems that for fine-grained scientific retrieval, the sbert embeddings (`all-mpnet-base-v2`) are better.
2. One can boost the performance of the off-the-shelf embeddings pretty significantly by training a linear projection on top of them.

|                                                                                     |   score_1 |   score_5 |   score_10 |   dim |   effective_dim |
|:------------------------------------------------------------------------------------|----------:|----------:|-----------:|------:|----------------:|
| ('sfn_2015_subsample.csv', 'sentence-transformers/all-mpnet-base-v2', '1580415070') |     1.183 |     1.084 |      1.027 |   512 |           2.477 |
| ('sfn_2015_subsample.csv', 'sentence-transformers/all-mpnet-base-v2', '1150734315') |     1.175 |     1.08  |      1.024 |    64 |           2.496 |
| ('sfn_2015_subsample.csv', 'sentence-transformers/all-mpnet-base-v2', '1795493411') |     1.174 |     1.083 |      1.024 |  1024 |           2.456 |
| ('sfn_2015_subsample.csv', 'sentence-transformers/all-mpnet-base-v2', '276397127')  |     1.176 |     1.086 |      1.024 |   128 |           2.525 |
| ('sfn_2015_subsample.csv', 'sentence-transformers/all-mpnet-base-v2', '1817992910') |     1.178 |     1.082 |      1.021 |   768 |           2.449 |
| ('sfn_2015_subsample.csv', 'sentence-transformers/all-mpnet-base-v2', '360739329')  |     1.167 |     1.083 |      1.018 |  2048 |           2.397 |
| ('sfn_2015_subsample.csv', 'sentence-transformers/all-mpnet-base-v2', '508533280')  |     1.169 |     1.082 |      1.017 |   256 |           2.407 |
| ('sfn_2015_subsample.csv', 'sentence-transformers/all-mpnet-base-v2', '')           |     1.15  |     1.027 |      0.944 |   768 |           3.761 |
| ('sfn_2015_subsample.csv', 'sentence-transformers/all-MiniLM-L6-v2', '')            |     1.152 |     0.992 |      0.9   |   384 |           3.908 |
| ('sfn_2015_subsample.csv', 'hkunlp/instructor-xl', '')                              |     1.148 |     0.996 |      0.898 |   768 |           1.317 |
| ('sfn_2015_subsample.csv', 'hkunlp/instructor-large', '')                           |     1.133 |     0.964 |      0.883 |   768 |           1.214 |
| ('sfn_2015_subsample.csv', 'sentence-transformers/allenai-specter', '')             |     1.04  |     0.917 |      0.849 |   768 |           1.556 |
| ('sfn_2015_subsample.csv', 'text-embedding-ada-002', '')                            |     1.026 |     0.918 |      0.845 |  1536 |           1.276 |
| ('sfn_2015_subsample.csv', 'intfloat/e5-base', '')                                  |     1.072 |     0.905 |      0.831 |   768 |           1.232 |
| ('sfn_2015_subsample.csv', 'sentence-transformers/gtr-t5-large', '')                |     0.979 |     0.789 |      0.717 |   768 |           1.38  |
| ('sfn_2015_subsample.csv', 'intfloat/e5-large', '')                                 |     0.302 |     0.251 |      0.261 |  1024 |           1     |

# Setup instructions

To install the local package:

```
pip install -e .
```

To run:

```
make data/processed/scores.csv
```

Visualize the results in `scripts/Compile Results.ipynb`.
