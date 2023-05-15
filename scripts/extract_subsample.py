import pandas as pd


def main():
    df = pd.read_csv("data/raw/sfn_2015.csv")
    df = df.sample(frac=0.1)
    df.to_csv("data/interim/sfn_2015_subsample.csv", index=False)


if __name__ == "__main__":
    main()
