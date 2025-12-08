"""Simple train/test split for the SEER dataset.

Hard-coded paths and split parameters; no CLI arguments.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# Configuration
INPUT_CSV = Path("./rag/seer_data_cleaned.csv")
TRAIN_CSV = Path("./rag/seer_data_train.csv")
TEST_CSV = Path("./rag/seer_data_test.csv")
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main() -> None:
	df = pd.read_csv(INPUT_CSV)

	train_df, test_df = train_test_split(
		df,
		test_size=TEST_SIZE,
		random_state=RANDOM_STATE,
		shuffle=True,
	)

	TRAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
	TEST_CSV.parent.mkdir(parents=True, exist_ok=True)

	train_df.to_csv(TRAIN_CSV, index=False)
	test_df.to_csv(TEST_CSV, index=False)

	print(f"Wrote train split: {len(train_df)} rows -> {TRAIN_CSV}")
	print(f"Wrote test split:  {len(test_df)} rows -> {TEST_CSV}")


if __name__ == "__main__":
	main()
