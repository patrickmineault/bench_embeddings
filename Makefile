data/interim/sfn_2015_subsample.csv: data/raw/sfn_2015.csv scripts/extract_subsample.py
	python scripts/extract_subsample.py

data/interim/sfn_2015_train.parquet data/interim/sfn_2015_pairs.parquet: data/interim/sfn_2015_subsample.csv data/sfn_2015.csv scripts/compile_reranking.py
	python scripts/compile_reranking.py

data/processed/hyperparameter_search.parquet: data/interim/sfn_2015_pairs.parquet scripts/train_reranking.py
	python scripts/train_reranking.py

data/interim/sfn_2015_sentence-transformers-all-mpnet-base-v2_all.pkl: data/processed/hyperparameter_search.parquet scripts/embed.py
	python scripts/embed.py

data/processed/scores.csv: data/processed/hyperparameter_search.parquet data/interim/sfn_2015_sentence-transformers-all-mpnet-base-v2_all.pkl scripts/benchmark.py
	python scripts/benchmark.py