import pickle
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch


class EmbeddingModel(torch.nn.Module):
    def __init__(
        self,
        embedding_length: int,
        modified_embedding_length: int,
        dropout_fraction: float = 0.0,
    ):
        super(EmbeddingModel, self).__init__()
        self.projection_matrix = torch.nn.Parameter(
            torch.randn(embedding_length, modified_embedding_length)
            / np.sqrt(modified_embedding_length)
        )
        self.dropout = torch.nn.Dropout(p=dropout_fraction)

    def forward(self, x):
        return self.dropout(x) @ self.projection_matrix


def optimize_matrix(
    modified_embedding_length: int = 2048,
    batch_size: int = 100,
    max_epochs: int = 100,
    learning_rate: float = 100.0,
    dropout_fraction: float = 0.0,
    df: pd.DataFrame = None,
    print_progress: bool = True,
    save_results: bool = True,
    device: str = "cuda",
) -> torch.tensor:
    """Return matrix optimized to minimize loss on training data."""
    run_id = random.randint(0, 2**31 - 1)  # (range is arbitrary)

    # convert from dataframe to torch tensors
    # e is for embedding, s for similarity label
    def tensors_from_dataframe(
        df: pd.DataFrame,
        embedding_column_1: str,
        embedding_column_2: str,
        similarity_label_column: str,
    ) -> Tuple[torch.tensor]:
        e1 = np.stack(np.array(df[embedding_column_1].values))
        e2 = np.stack(np.array(df[embedding_column_2].values))
        s = np.stack(
            np.array(df[similarity_label_column].astype("float").values)
        )

        e1 = torch.from_numpy(e1).float()
        e2 = torch.from_numpy(e2).float()
        s = torch.from_numpy(s).float()

        return e1, e2, s

    e1_train, e2_train, s_train = tensors_from_dataframe(
        df[df["dataset"] == "train"],
        "text_1_embedding",
        "text_2_embedding",
        "label",
    )
    e1_test, e2_test, s_test = tensors_from_dataframe(
        df[df["dataset"] == "test"],
        "text_1_embedding",
        "text_2_embedding",
        "label",
    )

    assert len(e1_train) > len(e1_test)

    # create dataset and loader
    dataset = torch.utils.data.TensorDataset(e1_train, e2_train, s_train)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # define loss function to minimize
    def mse_loss(predictions, targets):
        difference = predictions - targets
        return torch.sum(difference * difference) / difference.numel()

    # initialize projection matrix
    embedding_length = len(df["text_1_embedding"].values[0])

    model = EmbeddingModel(
        embedding_length, modified_embedding_length, dropout_fraction
    )
    model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epochs, types, losses, accuracies = [], [], [], []
    for epoch in range(1, 1 + max_epochs):
        # iterate through training dataloader
        model.train(True)
        for a, b, actual_similarity in train_loader:
            # generate prediction
            modified_embedding_1 = model(a.to(device=device))
            modified_embedding_2 = model(b.to(device=device))

            predicted_similarity = torch.nn.functional.cosine_similarity(
                modified_embedding_1, modified_embedding_2
            )
            # get loss and perform backpropagation
            loss = mse_loss(
                predicted_similarity, actual_similarity.to(device=device)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.train(False)
        # calculate test loss
        modified_embedding_1 = model(e1_test.to(device=device))
        modified_embedding_2 = model(e2_test.to(device=device))

        predicted_similarity = torch.nn.functional.cosine_similarity(
            modified_embedding_1, modified_embedding_2
        )

        test_loss = mse_loss(predicted_similarity, s_test.to(device=device))

        # calculate test accuracy
        for dataset in ["train", "test"]:
            # record results of each epoch
            epochs.append(epoch)
            types.append(dataset)
            losses.append(
                loss.item() if dataset == "train" else test_loss.item()
            )
            accuracies.append(a)

            # optionally print accuracies
            if print_progress is True:
                print(
                    f"Epoch {epoch}/{max_epochs}: {dataset} loss: {losses[-1]:.4}"
                )

    data = pd.DataFrame(
        {
            "epoch": epochs,
            "type": types,
            "loss": losses,
        }
    )
    data["run_id"] = run_id
    data["modified_embedding_length"] = modified_embedding_length
    data["batch_size"] = batch_size
    data["max_epochs"] = max_epochs
    data["learning_rate"] = learning_rate
    data["dropout_fraction"] = dropout_fraction
    if save_results is True:
        data.to_csv(
            f"data/processed/{run_id}_optimization_results.csv", index=False
        )
        with open(f"data/processed/{run_id}_matrix.pkl", "wb") as f:
            pickle.dump(model.projection_matrix.cpu().detach().numpy(), f)

    return data


def main():
    # example hyperparameter search
    # I recommend starting with max_epochs=10 while initially exploring
    df = pd.read_parquet("data/interim/sfn_2015_pairs.parquet")
    print(df.shape)
    df = df.rename(columns={"fold": "dataset"})

    results = []
    max_epochs = 30
    dropout_fraction = 0.2
    lr = 1e-1
    for learning_rate, batch_size, modified_embedding_length in [
        (lr, 512, 64),
        (lr, 512, 128),
        (lr, 512, 256),
        (lr, 512, 512),
        (lr, 512, 768),
        (lr, 512, 1024),
        (lr, 512, 2048),
    ]:
        result = optimize_matrix(
            modified_embedding_length=modified_embedding_length,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            dropout_fraction=dropout_fraction,
            save_results=True,
            df=df,
        )
        results.append(result)

    df = pd.concat(results, axis=0)
    df.to_parquet("data/processed/hyperparameter_search.parquet", index=False)


if __name__ == "__main__":
    main()
