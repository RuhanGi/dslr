import pandas as pd
import sys
import os

def stratified_split(input_file, test_ratio=0.2):
	try:
		# Load data
		df = pd.read_csv(input_file)

		# Check for required column
		if 'Hogwarts House' not in df.columns:
			print("Error: 'Hogwarts House' column not found.")
			return

		train_df = pd.DataFrame()
		test_df = pd.DataFrame()

		# Group by Hogwarts House for stratification
		for house, group in df.groupby('Hogwarts House'):
			test_size = int(len(group) * test_ratio)
			shuffled = group.sample(frac=1, random_state=42)
			test_group = shuffled.iloc[:test_size]
			train_group = shuffled.iloc[test_size:]
			test_df = pd.concat([test_df, test_group])
			train_df = pd.concat([train_df, train_group])

		# Shuffle final datasets
		train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
		test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

		# Save to CSV
		train_df.to_csv("datasets/ex_train.csv", index=False)
		test_df.to_csv("datasets/ex_test.csv", index=False)
		print("✅ Split complete: ex_train.csv and ex_test.csv created.")
	except Exception as e:
		print("Error:", e)

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Usage: python split_stratified.py <filename.csv>")
		sys.exit(1)

	input_file = sys.argv[1]
	stratified_split(input_file)
