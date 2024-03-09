import argparse

from data import load_dataset
from preprocessing import get_train_val_data
from config import DATA

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Load dataset and preprocess it.")
    parser.add_argument("--traindata", type=str, default="True", help="Create train and val dataset.")

    args = parser.parse_args()

    # Load dataset
    load_dataset()
    # Create train and val dataset if needed
    if args.traindata.lower() == "true":
        get_train_val_data(DATA["input"], DATA["train"], DATA["val"])