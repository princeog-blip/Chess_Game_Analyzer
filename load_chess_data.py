import os
import sys
import pandas as pd
from datasets import load_dataset

print("Connecting to Hugging Face...")

try:
    dataset = load_dataset(
        "Lichess/chess-position-evaluations",
        split="train",
        streaming=True
    )

    print("Loading chess positions...")

    # Set the number of records to fetch (e.g. 100 or 1000)
    sample = list(dataset.take(50000))
    df = pd.DataFrame(sample)

    print(df.head())

    df.to_csv("chess_data.csv", index=False)
    print("Dataset saved successfully!")

except Exception as e:
    print("Error loading dataset:", e)

# Flush stdout/stderr before exiting
sys.stdout.flush()
sys.stderr.flush()

# Cleanly exit immediately to avoid Windows socket cleanup warnings on teardown
os._exit(0)
